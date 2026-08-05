import builtins

import pytest

from truss.util.notebook import is_notebook_or_ipython


@pytest.mark.parametrize(
    "shell_class_name, expected",
    [
        ("ZMQInteractiveShell", True),
        ("TerminalInteractiveShell", True),
        ("SomeOtherShell", False),
    ],
)
def test_is_notebook_or_ipython_by_shell_type(monkeypatch, shell_class_name, expected):
    shell = type(shell_class_name, (), {})()
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell, raising=False)

    assert is_notebook_or_ipython() is expected


def test_is_notebook_or_ipython_in_plain_interpreter(monkeypatch):
    monkeypatch.delattr(builtins, "get_ipython", raising=False)

    assert is_notebook_or_ipython() is False
