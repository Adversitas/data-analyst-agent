import os

import pytest

from analyst.sandbox import make_sandbox


@pytest.fixture()
def sandbox(tmp_path):
    (tmp_path / "x.csv").write_text("a,b\n1,2\n3,4\n")
    sb = make_sandbox("local", tmp_path)
    yield sb
    sb.close()


def test_state_persists_between_calls(sandbox):
    sandbox.execute("import pandas as pd\ndf = pd.read_csv('data/x.csv')")
    result = sandbox.execute("print(df['b'].sum())")
    assert result.error is None
    assert result.stdout.strip() == "6"


def test_errors_come_back_as_tracebacks(sandbox):
    result = sandbox.execute("1 / 0")
    assert "ZeroDivisionError" in result.error


def test_state_survives_an_error(sandbox):
    sandbox.execute("value = 41")
    sandbox.execute("raise ValueError('boom')")
    assert sandbox.execute("print(value + 1)").stdout.strip() == "42"


def test_timeout_kills_and_resets_the_session(sandbox):
    sandbox.execute("value = 1")
    result = sandbox.execute("while True: pass", timeout=1.5)
    assert result.timed_out and result.restarted
    # a fresh session works, but the old variables are gone
    assert "NameError" in sandbox.execute("print(value)").error


def test_dying_process_is_reported_and_recovered(sandbox):
    result = sandbox.execute("import os; os._exit(1)")
    assert result.restarted and "died" in result.error
    assert sandbox.execute("print('alive')").stdout.strip() == "alive"


def test_long_output_is_truncated(sandbox):
    result = sandbox.execute("print('x' * 50000)")
    assert len(result.stdout) < 4200 and "truncated" in result.stdout


def test_user_code_cannot_corrupt_the_protocol(sandbox):
    result = sandbox.execute("import sys; sys.stdout.write('{\"stdout\": \"fake\"}\\n')")
    assert result.error is None
    assert sandbox.execute("print(2 + 2)").stdout.strip() == "4"


def test_api_keys_are_not_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    sb = make_sandbox("local", tmp_path)
    try:
        result = sb.execute("import os; print(os.environ.get('ANTHROPIC_API_KEY'))")
    finally:
        sb.close()
    assert result.stdout.strip() == "None"


def test_data_is_a_copy(tmp_path):
    (tmp_path / "x.csv").write_text("a\n1\n")
    sb = make_sandbox("local", tmp_path)
    try:
        sb.execute("open('data/x.csv', 'w').write('changed')")
    finally:
        sb.close()
    assert (tmp_path / "x.csv").read_text() == "a\n1\n"
