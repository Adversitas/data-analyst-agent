"""Isolation guarantees of the Docker sandbox. Skipped if Docker or the image is missing:
  docker build -t analyst-sandbox:latest -f docker/Dockerfile.sandbox docker
"""
import subprocess

import pytest

from analyst.sandbox import make_sandbox
from analyst.sandbox.docker import IMAGE


def _image_available() -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _image_available(), reason="Docker or the sandbox image is unavailable")


@pytest.fixture()
def sandbox(tmp_path):
    (tmp_path / "x.csv").write_text("a\n1\n")
    sb = make_sandbox("docker", tmp_path)
    yield sb
    sb.close()


def test_runs_code_and_keeps_state(sandbox):
    sandbox.execute("import pandas as pd; df = pd.read_csv('data/x.csv')")
    assert sandbox.execute("print(int(df['a'].sum()))").stdout.strip() == "1"


def test_no_network(sandbox):
    result = sandbox.execute("import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)")
    assert result.error and ("OSError" in result.error or "unreachable" in result.error)


def test_data_is_read_only(sandbox):
    assert "Read-only" in sandbox.execute("open('data/x.csv', 'w')").error


def test_root_filesystem_is_read_only(sandbox):
    assert "Read-only" in sandbox.execute("open('/work/evil.txt', 'w')").error


def test_tmp_is_writable_but_bounded(sandbox):
    assert sandbox.execute("open('/tmp/ok.txt', 'w').write('x')").error is None


def test_runs_as_non_root(sandbox):
    assert sandbox.execute("import os; print(os.getuid())").stdout.strip() == "10001"


def test_timeout_kills_the_container(sandbox):
    result = sandbox.execute("while True: pass", timeout=2)
    assert result.timed_out
    running = subprocess.run(["docker", "ps", "--filter", f"name={sandbox._name}", "-q"],
                             capture_output=True, text=True).stdout.strip()
    assert running == ""
