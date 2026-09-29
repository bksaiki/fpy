"""The scripts' options, as `--help` shows them, and what starting one loads."""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = ['local', 'perplexity', 'zeroshot', 'decode', 'chat']


class _Parsed(Exception):
    """Raised in place of parsing; `args[0]` is the parser a script built."""


def _parser(script: str, monkeypatch: pytest.MonkeyPatch) -> argparse.ArgumentParser:
    """The parser *script*'s `main` builds, stopped before it parses."""
    def stop(self: argparse.ArgumentParser, *args: object, **kwargs: object) -> None:
        raise _Parsed(self)

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', stop)
    with pytest.raises(_Parsed) as parsed:
        importlib.import_module(script).main([])
    return parsed.value.args[0]


@pytest.mark.parametrize('script', SCRIPTS)
def test_every_option_says_what_it_does_and_its_default(
    script: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each option has help, and one with a default names it."""
    for action in _parser(script, monkeypatch)._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        assert action.help, f'{script}: {action.dest} has no help'
        if action.default not in (None, False, argparse.SUPPRESS):
            assert 'default' in action.help, f'{script}: {action.dest} hides its default'


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('flag, shown', [
    (['--help'], 'options:'),
    (['--list-models'], 'kaitchup/Qwen3-0.6B-NVFP4'),
    (['--list-schemes'], 'fp8-block:fnuz'),
    (['--scheme', 'nvfp4', '--list-matmuls'], 'nv.blackwell.nvfp4'),
])
def test_help_or_a_list_prints_and_exits(
    script: str, flag: list[str], shown: str, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as done:
        importlib.import_module(script).main(flag)
    assert done.value.code == 0
    assert shown in capsys.readouterr().out


@pytest.mark.parametrize('script', SCRIPTS)
def test_a_script_starts_without_the_quantization_libraries(script: str) -> None:
    """`compressed_tensors` and `torchao` cost seconds at import; a script loads
    neither until a model does."""
    serve = Path(__file__).resolve().parents[1]
    code = (f'import sys; sys.path.insert(0, {str(serve)!r}); import {script}; '
            "print(*sorted({m.split('.')[0] for m in sys.modules} & {'compressed_tensors', 'torchao'}))")
    loaded = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, check=True).stdout.split()
    assert loaded == []
