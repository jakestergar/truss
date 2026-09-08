import logging
import threading
from typing import Dict, List

from truss.util.log_utils import LogInterceptor

logger = logging.getLogger(__name__)


def test_logs_emitted_inside_context_are_captured():
    with LogInterceptor() as interceptor:
        logger.warning("hello %s", "world")
        logger.error("boom")

    assert interceptor.get_logs() == ["hello world", "boom"]


def test_root_handlers_are_restored_on_exit():
    original_handlers = logging.root.handlers[:]

    with LogInterceptor() as interceptor:
        assert logging.root.handlers == [interceptor]

    assert logging.root.handlers == original_handlers


def test_root_handlers_are_restored_when_body_raises():
    original_handlers = logging.root.handlers[:]

    try:
        with LogInterceptor() as interceptor:
            logger.warning("before raise")
            raise RuntimeError("failure inside context")
    except RuntimeError:
        pass

    assert logging.root.handlers == original_handlers
    assert interceptor.get_logs() == ["before raise"]


def test_formatter_is_not_taken_from_original_root_handler():
    """Characterization test for a known quirk.

    The docstring of `LogInterceptor` claims it uses the formatter of the first
    root handler, but `_formatter` is read from `logging.root.handlers[0]`
    *after* the root handlers have already been replaced by `self`, so the first
    root handler at that point is the interceptor itself (whose formatter is
    None unless explicitly set). The original root handler's formatter is
    therefore never used. This pins current behavior; it is not a fix.
    """
    original_handlers = logging.root.handlers[:]
    prefixing_handler = logging.NullHandler()
    prefixing_handler.setFormatter(logging.Formatter("PREFIX: %(message)s"))
    logging.root.handlers = [prefixing_handler]

    try:
        with LogInterceptor() as interceptor:
            assert interceptor._formatter is None
            logger.warning("unprefixed")
    finally:
        logging.root.handlers = original_handlers

    assert interceptor.get_logs() == ["unprefixed"]


def test_threads_keep_their_logs_separate():
    start_barrier = threading.Barrier(2)
    logged_barrier = threading.Barrier(2)
    captured: Dict[str, List[str]] = {}

    def run(name: str) -> None:
        with LogInterceptor() as interceptor:
            start_barrier.wait(timeout=10)
            logger.warning("message from %s", name)
            logged_barrier.wait(timeout=10)
            captured[name] = interceptor.get_logs()

    threads = [
        threading.Thread(target=run, args=(name,)) for name in ("thread-a", "thread-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert captured == {
        "thread-a": ["message from thread-a"],
        "thread-b": ["message from thread-b"],
    }
