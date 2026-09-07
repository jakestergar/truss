import logging
import threading

import pytest

from truss.util.log_utils import LogInterceptor


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """Guarantees these tests cannot leak logging state into the rest of the suite."""
    root = logging.root
    original_handlers = root.handlers[:]
    original_level = root.level
    original_disabled = root.disabled
    original_thread_local = LogInterceptor._thread_local
    LogInterceptor._thread_local = threading.local()
    try:
        yield
    finally:
        LogInterceptor._thread_local = original_thread_local
        root.handlers = original_handlers
        root.level = original_level
        root.disabled = original_disabled


@pytest.fixture
def root_logger():
    logging.root.handlers = [logging.NullHandler()]
    logging.root.setLevel(logging.DEBUG)
    return logging.root


def test_captures_logs_and_restores_handlers(root_logger):
    original_handlers = root_logger.handlers[:]

    with LogInterceptor() as interceptor:
        assert root_logger.handlers == [interceptor]
        logging.info("hello")
        logging.warning("world")

    assert interceptor.get_logs() == ["hello", "world"]
    assert root_logger.handlers == original_handlers


def test_get_logs_before_and_after_context(root_logger):
    interceptor = LogInterceptor()
    assert interceptor.get_logs() == []

    with interceptor:
        logging.error("during")
        # Logs are readable while the context is still open.
        assert interceptor.get_logs() == ["during"]

    logging.error("after")
    assert interceptor.get_logs() == ["during"]


def test_handlers_restored_when_exception_propagates(root_logger):
    original_handlers = root_logger.handlers[:]
    interceptor = LogInterceptor()

    with pytest.raises(RuntimeError, match="boom"):
        with interceptor:
            logging.info("before raise")
            raise RuntimeError("boom")

    assert interceptor.get_logs() == ["before raise"]
    assert root_logger.handlers == original_handlers
    assert LogInterceptor._thread_local.handlers == []


def test_child_logger_propagation_and_levels(root_logger):
    child = logging.getLogger("truss.tests.log_utils.child")
    child.setLevel(logging.INFO)

    with LogInterceptor() as interceptor:
        child.debug("debug is filtered")
        child.info("info from child")
        logging.getLogger("truss.tests.log_utils.child.grandchild").warning("nested")

    assert interceptor.get_logs() == ["info from child", "nested"]


def test_root_level_filtering(root_logger):
    root_logger.setLevel(logging.WARNING)

    with LogInterceptor() as interceptor:
        logging.info("filtered out")
        logging.warning("kept")

    assert interceptor.get_logs() == ["kept"]


def test_percent_args_and_unicode(root_logger):
    with LogInterceptor() as interceptor:
        logging.warning("value is %s and %d", "ünïcode ✓", 42)

    assert interceptor.get_logs() == ["value is ünïcode ✓ and 42"]


def test_exc_info_is_included(root_logger):
    with LogInterceptor() as interceptor:
        try:
            raise ValueError("kaboom")
        except ValueError:
            logging.exception("failed")

    (message,) = interceptor.get_logs()
    assert message.startswith("failed\n")
    assert "ValueError: kaboom" in message
    assert "Traceback (most recent call last)" in message


def test_root_formatter_is_ignored_bug(root_logger):
    """BUG: the docstring promises the formatter of the first root handler is used.

    `__enter__` replaces `logging.root.handlers` with `[self]` *before* reading
    `logging.root.handlers[0].formatter`, so it always reads its own (unset)
    formatter and `self._formatter` stays `None`. Any format configured on the
    application's root handler (timestamps, level names, ...) is silently dropped
    from intercepted logs. Asserted as-is (current, wrong behavior).
    """
    handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("PREFIX %(levelname)s: %(message)s"))
    root_logger.handlers = [handler]

    with LogInterceptor() as interceptor:
        logging.warning("hello")

    assert interceptor._formatter is None
    # Would be "PREFIX WARNING: hello" if the documented behavior were implemented.
    assert interceptor.get_logs() == ["hello"]


def test_interceptors_own_formatter_is_picked_up(root_logger):
    # Corollary of the bug above: because `__enter__` reads
    # `logging.root.handlers[0].formatter` after installing itself, `self._formatter`
    # ends up being the interceptor's *own* formatter, never the application's.
    interceptor = LogInterceptor()
    own_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    interceptor.setFormatter(own_formatter)
    root_logger.handlers = [logging.NullHandler()]

    with interceptor:
        assert interceptor._formatter is own_formatter
        logging.warning("hello")

    assert interceptor.get_logs() == ["[WARNING] hello"]


def test_nested_interceptors(root_logger):
    original_handlers = root_logger.handlers[:]

    with LogInterceptor() as outer:
        logging.warning("outer only")
        with LogInterceptor() as inner:
            assert root_logger.handlers == [inner]
            logging.warning("inner only")
        assert root_logger.handlers == [outer]
        logging.warning("outer again")

    assert root_logger.handlers == original_handlers
    # The innermost interceptor swallows the records: the outer one never sees
    # messages logged while the inner context is open.
    assert outer.get_logs() == ["outer only", "outer again"]
    assert inner.get_logs() == ["inner only"]


def test_emit_routes_to_innermost_interceptor_not_self(root_logger):
    """`emit` appends to `_thread_local.handlers[-1]`, not to `self`.

    A handler that is installed on a specific logger (rather than on root) therefore
    donates its records to whichever interceptor is currently innermost.
    """
    donor = LogInterceptor()
    logger = logging.getLogger("truss.tests.log_utils.donor")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.handlers = [donor]
    try:
        with LogInterceptor() as active:
            logger.warning("logged via donor handler")
    finally:
        logger.handlers = []
        logger.propagate = True

    assert donor.get_logs() == []
    assert active.get_logs() == ["logged via donor handler"]


def test_reentering_same_instance_loses_original_handlers_bug(root_logger):
    """BUG: entering the same instance twice permanently clobbers root handlers.

    The second `__enter__` overwrites `self._original_handlers` with `[self]`, so
    both `__exit__` calls "restore" the interceptor itself. After the outermost
    `with` block the application's real root handlers are gone and the (now
    unmanaged) interceptor stays installed, which also makes subsequent logging
    raise `IndexError` because the thread-local stack is empty.
    """
    original_handlers = root_logger.handlers[:]
    interceptor = LogInterceptor()

    with interceptor:
        with interceptor:
            logging.warning("nested")

    assert root_logger.handlers == [interceptor]
    assert root_logger.handlers != original_handlers
    with pytest.raises(IndexError):
        logging.warning("logging is now broken")


def test_logs_stay_separated_between_threads(root_logger):
    results: dict[str, list[str]] = {}
    barrier = threading.Barrier(3)

    def worker(name: str) -> None:
        with LogInterceptor() as interceptor:
            barrier.wait()  # All interceptors are active concurrently.
            for i in range(5):
                logging.warning("%s-%d", name, i)
            barrier.wait()
            results[name] = interceptor.get_logs()

    threads = [
        threading.Thread(target=worker, args=(name,)) for name in ("a", "b", "c")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for name in ("a", "b", "c"):
        assert results[name] == [f"{name}-{i}" for i in range(5)]


def test_thread_without_interceptor_logs_are_dropped(root_logger):
    entered = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def other_thread() -> None:
        try:
            entered.wait(timeout=5)
            # Root handlers are globally replaced by the main thread's
            # interceptor, so this record is silently discarded.
            logging.warning("from thread without interceptor")
            done.set()
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    thread = threading.Thread(target=other_thread)
    thread.start()
    with LogInterceptor() as interceptor:
        entered.set()
        assert done.wait(timeout=5)
    thread.join()

    assert errors == []
    assert interceptor.get_logs() == []


def test_thread_with_exhausted_stack_raises_index_error_bug(root_logger):
    """BUG: logging from a thread that previously used an interceptor raises.

    Once a thread has entered and exited a `LogInterceptor`, its thread-local
    `handlers` list exists but is empty. If *another* thread has an interceptor
    installed on root at that moment, `emit` evaluates `handlers[-1]` on the empty
    list and the `IndexError` propagates out of the plain `logging.warning(...)`
    call in the innocent thread.
    """
    main_entered = threading.Event()
    worker_warmed_up = threading.Event()
    raised: list[BaseException] = []

    def worker() -> None:
        with LogInterceptor():
            logging.warning("warm up the thread-local stack")
        worker_warmed_up.set()
        main_entered.wait(timeout=5)
        try:
            logging.warning("boom")
        except BaseException as exc:
            raised.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    assert worker_warmed_up.wait(timeout=5)
    with LogInterceptor():
        main_entered.set()
        thread.join(timeout=5)

    assert len(raised) == 1
    assert isinstance(raised[0], IndexError)


def test_exiting_thread_clobbers_other_threads_interceptor_bug(root_logger):
    """BUG: `logging.root.handlers` is global but saved/restored per instance.

    Thread A enters, then thread B enters (saving A's handler as "original"), then
    A exits and restores the *pre-A* handlers. B is still inside its `with` block
    but no longer intercepts anything: its logs escape to the real root handlers
    and are missing from `get_logs()`.
    """
    a_entered = threading.Event()
    b_entered = threading.Event()
    a_exited = threading.Event()
    b_logs: list[str] = []

    def thread_a() -> None:
        with LogInterceptor():
            a_entered.set()
            b_entered.wait(timeout=5)
        a_exited.set()

    def thread_b() -> None:
        a_entered.wait(timeout=5)
        with LogInterceptor() as interceptor:
            b_entered.set()
            a_exited.wait(timeout=5)
            logging.warning("logged by b after a exited")
            b_logs.extend(interceptor.get_logs())

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    # The message is lost even though thread B's context manager is still active.
    assert b_logs == []
