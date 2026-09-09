import truss.util.notebook as notebook


def test_is_notebook_or_ipython_standard_interpreter():
    assert notebook.is_notebook_or_ipython() is False


def test_is_notebook_or_ipython_jupyter():
    class ZMQInteractiveShell:
        pass

    notebook.get_ipython = lambda: ZMQInteractiveShell()  # type: ignore[attr-defined]
    assert notebook.is_notebook_or_ipython() is True
    del notebook.get_ipython


def test_is_notebook_or_ipython_terminal():
    class TerminalInteractiveShell:
        pass

    notebook.get_ipython = lambda: TerminalInteractiveShell()  # type: ignore[attr-defined]
    assert notebook.is_notebook_or_ipython() is True
    del notebook.get_ipython


def test_is_notebook_or_ipython_other_shell():
    class OtherShell:
        pass

    notebook.get_ipython = lambda: OtherShell()  # type: ignore[attr-defined]
    assert notebook.is_notebook_or_ipython() is False
    del notebook.get_ipython
