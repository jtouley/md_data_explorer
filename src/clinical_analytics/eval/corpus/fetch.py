# SPDX-License-Identifier: MIT
"""Fetch a public corpus dataset into a cache outside the repo and package its upload.

Every file is checked against the sha256 pinned in ``corpus.yaml``; a mismatch stops the run.
A file already in the cache, or in a ``--seed`` directory (a corpus snapshot), is reused when
its checksum matches, so nothing is downloaded twice. A documentation file taken from a live
web page (``extract_section``) is stored and checked as the plain text of that one section, not
as the page bytes, which change with every site-template or view-count update. A seed file is
hard-linked into the cache when it can be, so a large snapshot is not stored twice.

The upload is packaged the way a user would upload it: a ZIP of the table and its documents, a
member taken out of the publisher's archive (``make: extract``, its own sha256 pinned), or one
file as it is (``upload: file``: a workbook, or a CSV uploaded on its own).

    uv run python -m clinical_analytics.eval.corpus.fetch covid_ms [--seed DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = structlog.get_logger()

CORPUS_YAML = Path(__file__).with_name("corpus.yaml")
CACHE_ENV = "MDDE_BENCH_CACHE"
DOWNLOAD_TIMEOUT_S = 120
# Fixed entry metadata so the same inputs give the same ZIP bytes on every platform: a fixed
# time, Unix as the creating system (zipfile writes 0 on Windows, 3 elsewhere), and no
# compression (DEFLATE output differs between zlib and zlib-ng builds of CPython).
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
ZIP_CREATE_SYSTEM_UNIX = 3

Downloader = Callable[[str], bytes]


class CorpusFetchError(RuntimeError):
    """A corpus file could not be fetched or packaged."""


class ChecksumMismatchError(CorpusFetchError):
    """A file's sha256 differs from the one pinned in corpus.yaml."""


class CorpusFile(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str
    url: str
    kind: str
    sha256: str
    # The id of an HTML <h2> section: the file is that section's plain text, and sha256 pins it.
    extract_section: str | None = None


class PackageEntry(BaseModel):
    """One file of the upload: ``from`` copied as fetched, or (``make: extract``) the archive
    ``from``'s ``member``, checked against its own pinned ``sha256``. ``a.zip!b.csv`` names
    ``b.csv`` inside the ZIP ``a.zip`` inside ``from`` (a ZIP in a ZIP, as UCI ships some sets)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name: str
    source: str = Field(alias="from")
    make: Literal["copy", "extract"] = "copy"
    member: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def _extract_is_pinned(self) -> PackageEntry:
        if self.make == "extract" and (self.member is None or self.sha256 is None):
            raise ValueError(f"{self.name}: make: extract needs a member and its sha256")
        if self.make == "copy" and self.member is not None:
            raise ValueError(f"{self.name}: a member is only read with make: extract")
        return self


class UploadPackage(BaseModel):
    """What the user uploads. ``zip``: the entries in one ZIP (``sha256`` pins the ZIP).
    ``file``: the one entry as it is (a workbook, or a CSV uploaded on its own)."""

    model_config = ConfigDict(extra="forbid")
    upload: Literal["zip", "file"] = "zip"
    zip: str | None = None
    contents: list[PackageEntry]
    sha256: str | None = None

    @model_validator(mode="after")
    def _shape_matches_upload(self) -> UploadPackage:
        if self.upload == "file" and (len(self.contents) != 1 or self.zip is not None):
            raise ValueError("upload: file takes exactly one entry and no zip name")
        if self.upload == "zip" and self.zip is None:
            raise ValueError("upload: zip needs a zip name")
        return self


class CorpusDataset(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    files: list[CorpusFile]
    upload_package: UploadPackage
    # The file the oracles read, in the dataset's files directory (an extracted member counts).
    # Default: the first ``kind: table`` file, else the uploaded file itself.
    oracle_input: str | None = None


def load_corpus(path: Path = CORPUS_YAML) -> dict[str, CorpusDataset]:
    """Datasets in corpus.yaml by name, validated."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = [CorpusDataset.model_validate(item) for item in raw["datasets"]]
    return {dataset.name: dataset for dataset in datasets}


def default_cache_dir() -> Path:
    """``$MDDE_BENCH_CACHE``, else ``~/.cache/mdde-bench``: never inside a checkout."""
    configured = os.environ.get(CACHE_ENV)
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "mdde-bench"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str) -> bytes:
    if not url.startswith("https://"):
        raise CorpusFetchError(f"refusing a non-HTTPS corpus URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "mdde-bench-fetch"})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310 - https only
        return bytes(response.read())


def _from_seed(path: str, sha256: str, dest: Path, seed_dirs: list[Path] | None) -> Path | None:
    """Place the first seed copy of ``path`` whose sha256 matches at ``dest``; the seed used.

    A hard link when the seed is on the same file system (no second copy of a large file),
    else a copy."""
    for seed in seed_dirs or []:
        candidate = seed / path
        if candidate.is_file() and sha256_file(candidate) == sha256:
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(candidate, dest)
            except OSError:
                shutil.copyfile(candidate, dest)
            return seed
    return None


def fetch_file(
    item: CorpusFile, dest_dir: Path, seed_dirs: list[Path] | None = None, download: Downloader = _download
) -> Path:
    """The checked file at ``dest_dir/item.path``: from the cache, a seed directory, or the URL.

    Raises:
        ChecksumMismatchError: a cached or downloaded file does not match the pinned sha256.
    """
    dest = dest_dir / item.path
    log = logger.bind(file=item.path)
    if dest.exists():
        actual = sha256_file(dest)
        if actual != item.sha256:
            raise ChecksumMismatchError(
                f"cached {dest} has sha256 {actual}, expected {item.sha256}: delete it and fetch again"
            )
        log.info("corpus_file_cached")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    seeded = _from_seed(item.path, item.sha256, dest, seed_dirs)
    if seeded is not None:
        log.info("corpus_file_seeded", seed=str(seeded))
        return dest
    body = download(item.url)
    if item.extract_section is not None:
        body = html_section_text(body, item.extract_section, item.url).encode("utf-8")
    actual = hashlib.sha256(body).hexdigest()
    if actual != item.sha256:
        raise ChecksumMismatchError(f"{item.url} returned sha256 {actual}, expected {item.sha256}")
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(body)
    partial.replace(dest)
    log.info("corpus_file_downloaded", bytes=len(body))
    return dest


def html_to_text(page: str, source_url: str) -> str:
    """Plain text of an official documentation page (format conversion only, no edits)."""
    text = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", page)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</h[1-6]>|</li>|</tr>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return f"Source: {source_url}\n(Plain-text rendering of the official project page.)\n\n{text}\n"


def html_section_text(page: bytes, section_id: str, source_url: str) -> str:
    """Plain text of one ``<h2 id=section_id>`` section of an HTML page, up to the next <h2>.

    Raises:
        CorpusFetchError: the page is not UTF-8 or has no such section.
    """
    try:
        markup = page.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorpusFetchError(f"{source_url} is not UTF-8 text: {error}") from error
    match = re.search(rf'(?is)<h2[^>]*\bid="{re.escape(section_id)}"[^>]*>.*?(?=<h2\b|$)', markup)
    if match is None:
        raise CorpusFetchError(f'{source_url} has no <h2 id="{section_id}"> section')
    return html_to_text(match.group(0), f"{source_url}#{section_id}")


def _member_bytes(archive_path: Path, member: str) -> bytes:
    """The bytes of ``member`` (``a.zip!b.csv`` for a ZIP inside the archive).

    Raises:
        CorpusFetchError: the archive or a nested ZIP has no such member.
    """
    body = archive_path.read_bytes()
    for part in member.split("!"):
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                body = archive.read(part)
        except (KeyError, zipfile.BadZipFile) as error:
            raise CorpusFetchError(f"{archive_path.name}: cannot read member {member!r}: {error}") from error
    return body


def entry_bytes(dataset: CorpusDataset, entry: PackageEntry, files_dir: Path) -> bytes:
    """One package entry's bytes. An extracted member is checked against its pinned sha256 and
    also written to ``files_dir/<name>``, where the oracles read it.

    Raises:
        CorpusFetchError: the entry names a file the dataset does not list.
        ChecksumMismatchError: an extracted member differs from its pinned sha256.
    """
    if entry.source not in {item.path for item in dataset.files}:
        raise CorpusFetchError(f"{dataset.name}: package entry {entry.name} names unknown file {entry.source}")
    if entry.make == "copy":
        return (files_dir / entry.source).read_bytes()
    body = _member_bytes(files_dir / entry.source, entry.member or "")
    actual = hashlib.sha256(body).hexdigest()
    if actual != entry.sha256:
        raise ChecksumMismatchError(f"{entry.source}!{entry.member} has sha256 {actual}, expected {entry.sha256}")
    (files_dir / entry.name).write_bytes(body)
    return body


def build_zip(dataset: CorpusDataset, files_dir: Path, out: Path) -> Path:
    """Write the upload ZIP: entries sorted by name, stored uncompressed with fixed metadata, so
    its bytes (and sha256) depend only on the checked inputs, not on the platform or zlib.

    Raises:
        ChecksumMismatchError: the ZIP does not match ``upload_package.sha256`` when one is pinned.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for entry in sorted(dataset.upload_package.contents, key=lambda e: e.name):
            info = zipfile.ZipInfo(entry.name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = ZIP_CREATE_SYSTEM_UNIX
            info.external_attr = 0o644 << 16
            archive.writestr(info, entry_bytes(dataset, entry, files_dir))
    body = buffer.getvalue()
    pinned = dataset.upload_package.sha256
    actual = hashlib.sha256(body).hexdigest()
    if pinned is not None and actual != pinned:
        raise ChecksumMismatchError(f"{dataset.name} upload ZIP has sha256 {actual}, expected {pinned}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    logger.info("corpus_zip_built", dataset=dataset.name, sha256=actual, bytes=len(body))
    return out


def build_upload(dataset: CorpusDataset, root: Path) -> Path:
    """The file to upload, by ``upload_package.upload``: the built ZIP, or the one entry as it is.

    Raises:
        CorpusFetchError: an entry names a file the dataset does not list, or a missing member.
        ChecksumMismatchError: the ZIP or an extracted member differs from its pinned sha256.
    """
    package, files_dir = dataset.upload_package, root / "files"
    if package.upload == "file":
        entry = package.contents[0]
        out = root / entry.name
        out.write_bytes(entry_bytes(dataset, entry, files_dir))
        return out
    # UploadPackage's validator requires a zip name for upload: zip.
    return build_zip(dataset, files_dir, root / str(package.zip))


def fetch_dataset(
    name: str,
    cache_dir: Path | None = None,
    seed_dirs: list[Path] | None = None,
    download: Downloader = _download,
    corpus: dict[str, CorpusDataset] | None = None,
) -> Path:
    """Fetch and check every file of ``name``, then build its upload; returns the upload's path."""
    datasets = corpus if corpus is not None else load_corpus()
    if name not in datasets:
        raise CorpusFetchError(f"unknown corpus dataset '{name}'; known: {', '.join(sorted(datasets))}")
    dataset = datasets[name]
    root = (cache_dir or default_cache_dir()) / name
    for item in dataset.files:
        fetch_file(item, root / "files", seed_dirs, download)
    return build_upload(dataset, root)


def oracle_input(dataset: CorpusDataset, upload: Path) -> Path:
    """The file the oracles read: ``oracle_input``, else the first table, else the upload."""
    files_dir = upload.parent / "files"
    if dataset.oracle_input is not None:
        return files_dir / dataset.oracle_input
    table = next((item.path for item in dataset.files if item.kind == "table"), None)
    return files_dir / table if table is not None else upload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--cache", type=Path, default=None, help=f"cache dir (default ${CACHE_ENV} or ~/.cache)")
    parser.add_argument("--seed", type=Path, action="append", default=[], help="directory of already-fetched files")
    args = parser.parse_args(argv)
    try:
        print(fetch_dataset(args.dataset, args.cache, args.seed))
    except CorpusFetchError as error:
        print(f"fetch failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
