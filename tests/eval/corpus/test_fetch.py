"""Corpus fetcher: checksums are enforced, caches and seeds are reused, ZIPs are deterministic."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from clinical_analytics.eval.corpus.fetch import (
    ChecksumMismatchError,
    CorpusFetchError,
    UploadPackage,
    build_zip,
    default_cache_dir,
    fetch_dataset,
    fetch_file,
    load_corpus,
    oracle_input,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestFetchFile:
    def test_fetchFile_matchingDownload_writesCheckedFile(self, tmp_path, make_corpus_dataset, make_downloader):
        # Arrange
        dataset, urls = make_corpus_dataset()
        download, calls = make_downloader(urls)

        # Act
        path = fetch_file(dataset.files[0], tmp_path, download=download)

        # Assert
        assert sha256_file(path) == dataset.files[0].sha256
        assert calls == [dataset.files[0].url]

    def test_fetchFile_downloadChecksumMismatch_raisesAndWritesNothing(
        self, tmp_path, make_corpus_dataset, make_downloader
    ):
        # Arrange
        dataset, urls = make_corpus_dataset()
        download, _ = make_downloader({**urls, dataset.files[0].url: b"tampered"})

        # Act / Assert
        with pytest.raises(ChecksumMismatchError, match="expected"):
            fetch_file(dataset.files[0], tmp_path, download=download)
        assert list(tmp_path.iterdir()) == []

    def test_fetchFile_cachedFileMismatch_raisesWithoutDownload(self, tmp_path, make_corpus_dataset, make_downloader):
        # Arrange
        dataset, urls = make_corpus_dataset()
        download, calls = make_downloader(urls)
        (tmp_path / dataset.files[0].path).write_bytes(b"stale")

        # Act / Assert
        with pytest.raises(ChecksumMismatchError, match="delete it"):
            fetch_file(dataset.files[0], tmp_path, download=download)
        assert calls == []

    @pytest.mark.parametrize(
        ("seed_bytes", "expected_calls"),
        [pytest.param(None, 0, id="matching_seed_reused"), pytest.param(b"other", 1, id="mismatched_seed_ignored")],
    )
    def test_fetchFile_seedDirectory_reusedOnlyWhenChecksumMatches(
        self, tmp_path, make_corpus_dataset, make_downloader, seed_bytes, expected_calls
    ):
        # Arrange
        dataset, urls = make_corpus_dataset()
        download, calls = make_downloader(urls)
        seed = tmp_path / "seed"
        seed.mkdir()
        item = dataset.files[0]
        (seed / item.path).write_bytes(seed_bytes if seed_bytes is not None else urls[item.url])

        # Act
        path = fetch_file(item, tmp_path / "cache", [seed], download=download)

        # Assert
        assert sha256_file(path) == item.sha256
        assert len(calls) == expected_calls

    def test_fetchFile_livePageChanged_sectionTextStillMatches(self, tmp_path, make_corpus_dataset, make_downloader):
        # Arrange: the page's other sections changed since the pin; the dictionary did not.
        dataset, urls = make_corpus_dataset()
        item = dataset.files[1]
        changed = urls[item.url].replace(b"Views: 12", b"Views: 13") + b"<script>csrf=abc</script>"
        download, _ = make_downloader({**urls, item.url: changed})

        # Act
        path = fetch_file(item, tmp_path, download=download)

        # Assert
        assert sha256_file(path) == item.sha256
        assert "Views" not in path.read_text(encoding="utf-8")

    def test_fetchFile_sectionMissing_raisesFetchError(self, tmp_path, make_corpus_dataset, make_downloader):
        # Arrange
        dataset, urls = make_corpus_dataset()
        item = dataset.files[1]
        download, _ = make_downloader({**urls, item.url: b"<h2 id='other'>x</h2>"})

        # Act / Assert
        with pytest.raises(CorpusFetchError, match="no <h2"):
            fetch_file(item, tmp_path, download=download)


class TestBuildZip:
    def test_buildZip_sameInputsTwice_identicalBytesWithDictionaryText(
        self, tmp_path, make_corpus_dataset, make_downloader
    ):
        # Arrange
        dataset, urls = make_corpus_dataset()
        download, _ = make_downloader(urls)
        corpus = {dataset.name: dataset}

        # Act
        first = fetch_dataset(dataset.name, tmp_path / "a", download=download, corpus=corpus).read_bytes()
        second = fetch_dataset(dataset.name, tmp_path / "b", download=download, corpus=corpus).read_bytes()

        # Assert
        assert first == second
        with zipfile.ZipFile(tmp_path / "a" / dataset.name / "synthetic.zip") as archive:
            assert archive.namelist() == ["data_dictionary.txt", "table.csv"]
            assert "sex: biological sex" in archive.read("data_dictionary.txt").decode()

    @pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
    def test_buildZip_anyPlatform_sameBytesStoredUnixEntries(
        self, tmp_path, monkeypatch, make_corpus_dataset, make_downloader, platform
    ):
        # Arrange
        dataset, urls = make_corpus_dataset()
        download, _ = make_downloader(urls)
        reference = fetch_dataset(dataset.name, tmp_path / "ref", download=download, corpus={dataset.name: dataset})
        monkeypatch.setattr(zipfile.sys, "platform", platform)

        # Act
        built = build_zip(dataset, tmp_path / "ref" / dataset.name / "files", tmp_path / f"{platform}.zip")

        # Assert
        assert built.read_bytes() == reference.read_bytes()
        with zipfile.ZipFile(built) as archive:
            for info in archive.infolist():
                assert (info.create_system, info.compress_type) == (3, zipfile.ZIP_STORED)

    def test_buildZip_pinnedChecksumMismatch_raises(self, tmp_path, make_corpus_dataset, make_downloader):
        # Arrange
        dataset, urls = make_corpus_dataset(sha256="0" * 64)
        download, _ = make_downloader(urls)
        for item in dataset.files:
            fetch_file(item, tmp_path, download=download)

        # Act / Assert
        with pytest.raises(ChecksumMismatchError, match="upload ZIP"):
            build_zip(dataset, tmp_path, tmp_path / "out.zip")
        assert not (tmp_path / "out.zip").exists()

    def test_buildZip_pinnedChecksumMatches_writesZip(self, tmp_path, make_corpus_dataset, make_downloader):
        # Arrange
        dataset, urls = make_corpus_dataset()
        download, _ = make_downloader(urls)
        built = fetch_dataset(dataset.name, tmp_path / "a", download=download, corpus={dataset.name: dataset})
        pinned, _ = make_corpus_dataset(sha256=hashlib.sha256(built.read_bytes()).hexdigest())

        # Act
        out = build_zip(pinned, tmp_path / "a" / dataset.name / "files", tmp_path / "pinned.zip")

        # Assert
        assert out.read_bytes() == built.read_bytes()


class TestPackageEntries:
    def test_fetchDataset_memberOfNestedZip_uploadsCheckedMemberAndOracleReadsIt(
        self, tmp_path, make_archive_dataset, make_downloader
    ):
        # Arrange
        dataset, urls, table = make_archive_dataset()
        download, _ = make_downloader(urls)

        # Act
        upload = fetch_dataset(dataset.name, tmp_path, download=download, corpus={dataset.name: dataset})

        # Assert
        assert upload.name == "table.csv"
        assert upload.read_bytes() == table
        assert oracle_input(dataset, upload).read_bytes() == table

    def test_fetchDataset_memberChecksumMismatch_raises(self, tmp_path, make_archive_dataset, make_downloader):
        # Arrange
        dataset, urls, _ = make_archive_dataset(member_sha256="0" * 64)
        download, _ = make_downloader(urls)

        # Act / Assert
        with pytest.raises(ChecksumMismatchError, match="inner.zip!table.csv"):
            fetch_dataset(dataset.name, tmp_path, download=download, corpus={dataset.name: dataset})

    def test_fetchDataset_memberMissing_raisesFetchError(self, tmp_path, make_archive_dataset, make_downloader):
        # Arrange
        dataset, urls, _ = make_archive_dataset()
        dataset.upload_package.contents[0].member = "inner.zip!absent.csv"
        download, _ = make_downloader(urls)

        # Act / Assert
        with pytest.raises(CorpusFetchError, match="absent.csv"):
            fetch_dataset(dataset.name, tmp_path, download=download, corpus={dataset.name: dataset})

    @pytest.mark.parametrize(
        "package",
        [
            pytest.param(
                {
                    "contents": [{"name": "t.csv", "from": "a.zip", "make": "extract", "member": "t.csv"}],
                    "zip": "x.zip",
                },
                id="extract_without_sha256",
            ),
            pytest.param(
                {"contents": [{"name": "t.csv", "from": "t.csv", "member": "t.csv"}], "zip": "x.zip"},
                id="member_on_copy",
            ),
            pytest.param({"upload": "file", "contents": []}, id="file_without_entry"),
            pytest.param(
                {"upload": "file", "contents": [{"name": "a", "from": "a"}, {"name": "b", "from": "b"}]},
                id="file_with_two_entries",
            ),
            pytest.param({"contents": [{"name": "a", "from": "a"}]}, id="zip_without_name"),
            pytest.param({"zip": "x.zip", "contents": []}, id="zip_without_entries"),
            pytest.param({"upload": "prebuilt", "zip": "x.zip", "recipe": "r"}, id="prebuilt_without_sha256"),
            pytest.param({"upload": "prebuilt", "zip": "x.zip", "sha256": "0" * 64}, id="prebuilt_without_recipe"),
        ],
    )
    def test_uploadPackage_inconsistentShape_rejected(self, package):
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            UploadPackage.model_validate(package)


class TestPrebuiltUpload:
    @staticmethod
    def _dataset(body: bytes, make_corpus_dataset):
        dataset, urls = make_corpus_dataset()
        dataset.upload_package = UploadPackage.model_validate(
            {"upload": "prebuilt", "zip": "set.zip", "sha256": hashlib.sha256(body).hexdigest(), "recipe": "snapshot"}
        )
        return dataset, urls

    def test_fetchDataset_prebuiltInSeed_linkedIntoCacheAndReturned(
        self, tmp_path, make_corpus_dataset, make_downloader
    ):
        # Arrange
        body = b"PK-synthetic-prebuilt"
        dataset, urls = self._dataset(body, make_corpus_dataset)
        download, _ = make_downloader(urls)
        seed = tmp_path / "seed"
        seed.mkdir()
        (seed / "set.zip").write_bytes(body)

        # Act
        upload = fetch_dataset(dataset.name, tmp_path / "cache", [seed], download, {dataset.name: dataset})

        # Assert
        assert upload.read_bytes() == body
        assert upload.parent == tmp_path / "cache" / dataset.name

    @pytest.mark.parametrize(
        ("seeded", "cached", "error"),
        [
            pytest.param(None, None, CorpusFetchError, id="in_no_seed"),
            pytest.param(b"other", None, CorpusFetchError, id="seed_differs"),
            pytest.param(None, b"stale", ChecksumMismatchError, id="cache_differs"),
        ],
    )
    def test_fetchDataset_prebuiltMissingOrChanged_raises(
        self, tmp_path, make_corpus_dataset, make_downloader, seeded, cached, error
    ):
        # Arrange
        dataset, urls = self._dataset(b"PK-synthetic-prebuilt", make_corpus_dataset)
        download, _ = make_downloader(urls)
        seed = tmp_path / "seed"
        seed.mkdir()
        if seeded is not None:
            (seed / "set.zip").write_bytes(seeded)
        if cached is not None:
            (tmp_path / "cache" / dataset.name).mkdir(parents=True)
            (tmp_path / "cache" / dataset.name / "set.zip").write_bytes(cached)

        # Act / Assert
        with pytest.raises(error):
            fetch_dataset(dataset.name, tmp_path / "cache", [seed], download, {dataset.name: dataset})


class TestShippedCorpus:
    def test_loadCorpus_shippedYaml_pinsHttpsUrlsAndChecksums(self):
        # Arrange / Act
        corpus = load_corpus()

        # Assert
        assert "covid_ms" in corpus
        for dataset in corpus.values():
            package = dataset.upload_package
            if package.upload in ("zip", "prebuilt"):
                assert re.fullmatch(r"[0-9a-f]{64}", package.sha256 or "")
            for entry in package.contents:
                assert entry.make == "copy" or re.fullmatch(r"[0-9a-f]{64}", entry.sha256 or "")
            for item in dataset.files:
                assert item.url.startswith("https://")
                assert re.fullmatch(r"[0-9a-f]{64}", item.sha256)
        table = next(item for item in corpus["covid_ms"].files if item.kind == "table")
        assert table.sha256 == "8aafa48e4f5a4db9d1a15924317dce94745ea56b59e9b7cc7be2bfc4ecb97498"

    @pytest.mark.parametrize(
        ("env_value", "expected_parts"),
        [pytest.param(None, (".cache", "mdde-bench"), id="default"), pytest.param("bench", ("bench",), id="env")],
    )
    def test_defaultCacheDir_envOrHome_outsideRepo(self, monkeypatch, tmp_path, env_value, expected_parts):
        # Arrange
        monkeypatch.setenv("HOME", str(tmp_path))
        if env_value is None:
            monkeypatch.delenv("MDDE_BENCH_CACHE", raising=False)
        else:
            monkeypatch.setenv("MDDE_BENCH_CACHE", str(tmp_path / env_value))

        # Act
        cache = default_cache_dir()

        # Assert
        assert cache.parts[-len(expected_parts) :] == expected_parts
        assert not cache.resolve().is_relative_to(REPO_ROOT)
