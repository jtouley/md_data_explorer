#!/usr/bin/env python3
"""Smoke test the backend as an *installed* package, away from the repo checkout.

Steps:
1. ``uv build --wheel`` into a temp dir.
2. Fresh venv outside the repo; install the wheel plus the locked runtime dependencies.
3. Start ``python -m clinical_analytics.api.serve`` from an unrelated cwd, with HOME and
   XDG_DATA_HOME pointing at a temp data home and a per-launch API token. On Linux, when
   ``unshare -rn`` works, the server and this driver run in a network namespace with only
   loopback, so any attempt to reach the network fails.
4. GET /health (200), upload tests/fixtures/synthetic/cohort.csv, create a session, run a
   simple question and wait for it to complete.
5. Check the install prefix gained no files (the app writes only to the data home).

Synthetic data only; the driver never prints row-level values.

    uv run python scripts/smoke_installed.py            # or: make smoke-installed
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CSV = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "cohort.csv"
HEALTH_TIMEOUT_S = 180.0
QUERY_TIMEOUT_S = 120.0
SMOKE_QUESTION = "how many patients are there"


class SmokeError(RuntimeError):
    """The installed backend did not behave as required."""


@dataclass
class SmokeResult:
    health_status: int = 0
    upload_status: int = 0
    query_status: str = ""
    query_intent: str = ""
    offline: bool = False
    new_files_in_install: list[str] = field(default_factory=list)
    data_home_files: int = 0
    startup_seconds: float = 0.0
    embedding_model_loaded: bool = False
    golden_examples_embedded: bool = False
    server_log_tail: str = ""


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)  # type: ignore[call-overload]


def build_and_install(work: Path, extra_requirements: list[str] | None = None) -> Path:
    """Build the wheel and install it into ``work/venv``; return the venv's python."""
    dist = work / "dist"
    _run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=REPO_ROOT)
    wheels = sorted(dist.glob("clinical_analytics-*.whl"))
    if len(wheels) != 1:
        raise SmokeError(f"expected one wheel in {dist}, found {[w.name for w in wheels]}")
    reqs = work / "requirements.txt"
    export = _run(
        ["uv", "export", "--frozen", "--no-dev", "--no-emit-workspace", "--no-hashes", "--no-header"],
        cwd=REPO_ROOT,
    )
    reqs.write_text(export.stdout, encoding="utf-8")
    venv = work / "venv"
    _run(["uv", "venv", "--quiet", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", str(venv)])
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run(
        ["uv", "pip", "install", "--quiet", "--python", str(python), "-r", str(reqs), str(wheels[0])]
        + list(extra_requirements or [])
    )
    return python


def _package_dir(python: Path) -> Path:
    out = _run(
        [str(python), "-I", "-c", "import clinical_analytics, os; print(os.path.dirname(clinical_analytics.__file__))"],
        cwd=tempfile.gettempdir(),
    )
    return Path(out.stdout.strip())


def _snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if "__pycache__" not in p.parts}


def _bring_up_loopback() -> None:
    """In a fresh network namespace ``lo`` is down; bring it up without needing ``ip`` (Linux only)."""
    import fcntl  # POSIX-only; imported here so the module also loads on Windows
    siocgifflags, siocsifflags, iff_up = 0x8913, 0x8914, 0x1
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
        ifr = struct.pack("16sH14s", b"lo", 0, b"\0" * 14)
        flags = struct.unpack("16sH14s", fcntl.ioctl(sk, siocgifflags, ifr))[1]
        fcntl.ioctl(sk, siocsifflags, struct.pack("16sH14s", b"lo", flags | iff_up, b"\0" * 14))


def _network_reachable() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
    except OSError:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _request(
    method: str, url: str, token: str, body: bytes | None = None, ctype: str | None = None
) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"}
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - loopback only
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read()[:500].decode("utf-8", "replace")
        raise SmokeError(f"{method} {url} -> {e.code}: {detail}") from e


def _multipart(field_name: str, path: Path, extra: dict[str, str]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in extra.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        "Content-Type: text/csv\r\n\r\n".encode()
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def drive(python: Path, work: Path, offline: bool) -> SmokeResult:
    """Start the installed server from an unrelated cwd and exercise it (runs inside the netns)."""
    result = SmokeResult()
    if offline:
        _bring_up_loopback()
        if _network_reachable():
            raise SmokeError("network namespace still reaches the internet")
    result.offline = offline and not _network_reachable()

    home = work / "home"
    xdg = work / "xdg-data"
    cwd = work / "unrelated-cwd"
    for d in (home, xdg, cwd):
        d.mkdir(parents=True, exist_ok=True)
    package_dir = _package_dir(python)
    before = _snapshot(package_dir)

    token = secrets.token_urlsafe(32)
    port = _free_port()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "XDG_DATA_HOME": str(xdg),
        "CLINICAL_ANALYTICS_API_TOKEN": token,
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LANG": "C.UTF-8",
    }
    log_path = work / "server.log"
    base = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(  # noqa: S603 - our own venv python
            [str(python), "-m", "clinical_analytics.api.serve", "--port", str(port)],
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = started + HEALTH_TIMEOUT_S
        while True:
            if proc.poll() is not None:
                raise SmokeError(f"server exited with code {proc.returncode} before /health")
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:  # noqa: S310
                    result.health_status = resp.status
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if time.monotonic() > deadline:
                    raise SmokeError("timed out waiting for /health") from None
                time.sleep(0.5)
        result.startup_seconds = round(time.monotonic() - started, 2)

        body, ctype = _multipart("file", FIXTURE_CSV, {"dataset_name": "smoke_cohort"})
        status, upload = _request("POST", f"{base}/api/datasets/upload", token, body, ctype)
        result.upload_status = status
        dataset_id = upload.get("upload_id") or upload.get("dataset_id")
        if not dataset_id:
            raise SmokeError(f"upload returned no id (keys: {sorted(upload)})")

        _, session = _request(
            "POST", f"{base}/api/sessions", token, json.dumps({"dataset_id": dataset_id}).encode(), "application/json"
        )
        payload = {"session_id": session["session_id"], "dataset_id": dataset_id, "query_text": SMOKE_QUESTION}
        _, submitted = _request("POST", f"{base}/api/queries", token, json.dumps(payload).encode(), "application/json")
        query_id = submitted["query_id"]
        deadline = time.monotonic() + QUERY_TIMEOUT_S
        while True:
            _, qr = _request("GET", f"{base}/api/queries/{query_id}", token)
            result.query_status, result.query_intent = qr.get("status", ""), qr.get("intent", "")
            if result.query_status in {"completed", "failed", "awaiting_confirmation"} or time.monotonic() > deadline:
                break
            time.sleep(0.5)
    except SmokeError as e:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise SmokeError(f"{e}\n--- server log (tail) ---\n{tail}") from e
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        result.server_log_tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        result.embedding_model_loaded = "embedding_model_loaded" in log_text
        result.golden_examples_embedded = "golden_examples_loaded" in log_text

    result.new_files_in_install = sorted(_snapshot(package_dir) - before)
    result.data_home_files = sum(1 for p in xdg.rglob("*") if p.is_file())
    return result


def _unshare_available() -> bool:
    if sys.platform != "linux" or shutil.which("unshare") is None:
        return False
    return subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0


def run_smoke(work: Path, extra_requirements: list[str] | None = None) -> SmokeResult:
    """Build, install and drive; returns the result (raises SmokeError on hard failures)."""
    python = build_and_install(work, extra_requirements)
    offline = _unshare_available()
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--drive", str(python), str(work)]
    if offline:
        cmd = ["unshare", "-rn", *cmd, "--offline"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SmokeError(f"driver failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")
    return SmokeResult(**json.loads(proc.stdout.strip().splitlines()[-1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive", nargs=2, metavar=("PYTHON", "WORKDIR"), help=argparse.SUPPRESS)
    parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep", action="store_true", help="keep the temp work directory")
    args = parser.parse_args(argv)
    if args.drive:
        res = drive(Path(args.drive[0]), Path(args.drive[1]), args.offline)
        print(json.dumps(asdict(res)))
        return 0
    work = Path(tempfile.mkdtemp(prefix="ca-smoke-"))
    try:
        res = run_smoke(work)
        print(json.dumps({k: v for k, v in asdict(res).items() if k != "server_log_tail"}, indent=2))
        ok = res.health_status == 200 and res.query_status == "completed" and not res.new_files_in_install
        if not ok:
            print(res.server_log_tail, file=sys.stderr)
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
