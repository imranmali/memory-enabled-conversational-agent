import builtins
import io
import runpy
import sys

import pytest

from memory_agent.cli import main


def run(argv, text, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    code = main(argv)
    return code, capsys.readouterr()


def test_ci_smoke_scenario(tmp_path, monkeypatch, capsys):
    data = str(tmp_path / "mem")
    code, first = run(
        ["--user", "ada", "--data-dir", data],
        "My name is Ada and I love hiking.\n/quit\n",
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "assistant> Nice to meet you, Ada!" in first.out
    code, second = run(
        ["--user", "ada", "--data-dir", data],
        "What do you remember about me?\n/quit\n",
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert "hiking" in second.out.lower()
    assert "Here's what I remember about you" in second.out
    assert "you> " not in second.out  # no prompt when stdin is not a TTY


def test_commands(tmp_path, monkeypatch, capsys):
    script = "\n".join(
        [
            "I love hiking",
            "",
            "/help",
            "/facts",
            "/memories",
            "/memories hiking",
            "/memories zzzz qqqq",
            "/stats",
            "/sync",
            "/bogus",
            "/forget-all",
            "/facts",
            "/memories",
        ]
    )
    code, out = run(
        ["--user", "ada", "--data-dir", str(tmp_path), "--device", "pc"],
        script + "\n",
        monkeypatch,
        capsys,
    )
    lines = out.out
    assert code == 0
    assert "/forget-all" in lines  # help text
    assert "- User loves hiking" in lines
    assert "(fact) User loves hiking" in lines
    assert "] (fact) User loves hiking" in lines  # scored recall output
    assert "No memories found." in lines
    assert "device: pc" in lines and "facts: 1" in lines and "session messages: 2" in lines
    assert "Synced: pushed" in lines
    assert "Unknown command: /bogus" in lines
    assert "Forgot " in lines
    assert "I don't know any facts yet." in lines
    assert lines.rstrip().endswith("Goodbye! I'll remember our conversation.")  # EOF ends


def test_exit_alias_and_tty_prompt(tmp_path, monkeypatch, capsys):
    class TTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", TTY("hello\n/exit\nnever processed\n"))
    assert main(["--user", "ada", "--data-dir", str(tmp_path), "--device", "pc"]) == 0
    out = capsys.readouterr().out
    assert "you> " in out
    assert "never processed" not in out


def test_explicit_streams(tmp_path):
    out = io.StringIO()
    code = main(
        ["--user", "bo", "--data-dir", str(tmp_path), "--device", "pc"],
        stdin=io.StringIO("Call me Bo\n"),
        stdout=out,
    )
    assert code == 0
    assert "Nice to meet you, Bo!" in out.getvalue()


def test_missing_user_is_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_reserved_device_rejected(tmp_path, capsys):
    code = main(
        ["--user", "ada", "--data-dir", str(tmp_path), "--device", "local"],
        stdin=io.StringIO(""),
    )
    assert code == 2
    assert "device_id" in capsys.readouterr().err


def test_anthropic_without_sdk(tmp_path, monkeypatch, capsys):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = main(
        ["--user", "ada", "--data-dir", str(tmp_path), "--llm", "anthropic", "--model", "m"],
        stdin=io.StringIO(""),
    )
    assert code == 2
    assert "pip install" in capsys.readouterr().err


def test_python_dash_m(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["memory_agent", "--user", "ada", "--data-dir", str(tmp_path), "--device", "pc"],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("I love tea\n/quit\n"))
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("memory_agent", run_name="__main__")
    assert exc.value.code == 0
    assert "you love tea" in capsys.readouterr().out
