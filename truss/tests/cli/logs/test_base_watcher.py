import types
from typing import Any, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

from truss.cli.logs import base_watcher
from truss.cli.logs.base_watcher import (
    CLOCK_SKEW_BUFFER_MS,
    POLL_INTERVAL_SEC,
    LogWatcher,
)
from truss.cli.logs.utils import ParsedLog
from truss.remote.baseten.api import BasetenApi

NOW_MS = 1_700_000_000_000


def raw_log(message="hello", timestamp="1700000000000000000", replica="replica-1"):
    return {"timestamp": timestamp, "message": message, "replica": replica}


class FakeLogWatcher(LogWatcher):
    """Minimal concrete watcher that serves canned log batches."""

    def __init__(self, api, log_batches=None, poll_again=None):
        super().__init__(api)
        self.log_batches = list(log_batches or [])
        self.poll_again = list(poll_again or [])
        self.fetch_calls: List[tuple] = []
        self.before_polling_calls = 0
        self.after_polling_calls = 0
        self.post_poll_calls = 0

    def fetch_logs(
        self, start_epoch_millis: Optional[int], end_epoch_millis: Optional[int]
    ) -> List[Any]:
        self.fetch_calls.append((start_epoch_millis, end_epoch_millis))
        if not self.log_batches:
            return []
        return self.log_batches.pop(0)

    def before_polling(self) -> None:
        self.before_polling_calls += 1

    def after_polling(self) -> None:
        self.after_polling_calls += 1

    def should_poll_again(self) -> bool:
        return self.poll_again.pop(0) if self.poll_again else False

    def post_poll(self) -> None:
        self.post_poll_calls += 1


@pytest.fixture(autouse=True)
def isolate_log_hashes():
    # `_log_hashes` lives on the class, so state leaks between watchers.
    original = LogWatcher._log_hashes
    LogWatcher._log_hashes = set()
    yield
    LogWatcher._log_hashes = original


@pytest.fixture
def api():
    return Mock(spec=BasetenApi)


@pytest.fixture
def fake_time(monkeypatch):
    """Freeze `time.time` and record `time.sleep` calls in the module."""
    state = types.SimpleNamespace(now_ms=NOW_MS, sleeps=[])

    def _time():
        return state.now_ms / 1000

    def _sleep(seconds):
        state.sleeps.append(seconds)

    monkeypatch.setattr(
        base_watcher, "time", types.SimpleNamespace(time=_time, sleep=_sleep)
    )
    return state


def test_init_stores_api(api):
    assert FakeLogWatcher(api).api is api


def test_cannot_instantiate_abstract_watcher(api):
    with pytest.raises(TypeError):
        LogWatcher(api)  # type: ignore[abstract]


def test_hooks_default_to_no_ops(api):
    # The abstract hook bodies are no-ops; subclasses may call `super()`.
    watcher = FakeLogWatcher(api)
    assert LogWatcher.fetch_logs(watcher, None, None) is None
    assert LogWatcher.before_polling(watcher) is None
    assert LogWatcher.after_polling(watcher) is None
    assert LogWatcher.should_poll_again(watcher) is None
    assert LogWatcher.post_poll(watcher) is None


def test_hash_log_is_stable_for_equal_logs(api):
    watcher = FakeLogWatcher(api)
    log = ParsedLog(**raw_log())
    assert watcher._hash_log(log) == watcher._hash_log(ParsedLog(**raw_log()))


@pytest.mark.parametrize(
    "overrides",
    [
        {"message": "other"},
        {"timestamp": "1700000000000000001"},
        {"replica": "replica-2"},
        {"replica": None},
    ],
)
def test_hash_log_differs_when_any_field_differs(api, overrides):
    watcher = FakeLogWatcher(api)
    base = ParsedLog(**raw_log())
    other = ParsedLog(**raw_log(**overrides))
    assert watcher._hash_log(base) != watcher._hash_log(other)


def test_get_start_epoch_ms_is_none_before_first_poll(api):
    assert FakeLogWatcher(api).get_start_epoch_ms(NOW_MS) is None


def test_get_start_epoch_ms_subtracts_clock_skew_buffer(api):
    watcher = FakeLogWatcher(api)
    watcher._last_poll_time_ms = NOW_MS
    assert watcher.get_start_epoch_ms(NOW_MS + 1) == NOW_MS - CLOCK_SKEW_BUFFER_MS


def test_get_start_epoch_ms_treats_zero_last_poll_as_unset(api):
    # A falsy epoch (0) is indistinguishable from "never polled" here.
    watcher = FakeLogWatcher(api)
    watcher._last_poll_time_ms = 0
    assert watcher.get_start_epoch_ms(NOW_MS) is None


def test_fetch_and_parse_logs_yields_parsed_logs(api):
    watcher = FakeLogWatcher(api, log_batches=[[raw_log(message="a"), raw_log("b")]])
    logs = list(watcher.fetch_and_parse_logs(None, NOW_MS))
    assert [log.message for log in logs] == ["a", "b"]
    assert watcher.fetch_calls == [(None, NOW_MS)]


def test_fetch_and_parse_logs_handles_empty_response(api):
    watcher = FakeLogWatcher(api, log_batches=[[]])
    assert list(watcher.fetch_and_parse_logs(1, 2)) == []


def test_fetch_and_parse_logs_skips_duplicates_within_batch(api):
    watcher = FakeLogWatcher(api, log_batches=[[raw_log(), raw_log()]])
    assert len(list(watcher.fetch_and_parse_logs(None, NOW_MS))) == 1


def test_fetch_and_parse_logs_skips_logs_already_seen(api):
    watcher = FakeLogWatcher(
        api, log_batches=[[raw_log()], [raw_log(), raw_log("new")]]
    )
    list(watcher.fetch_and_parse_logs(None, NOW_MS))
    second = list(watcher.fetch_and_parse_logs(None, NOW_MS))
    assert [log.message for log in second] == ["new"]


def test_fetch_and_parse_logs_deduplicates_across_watchers(api):
    # Hashes are shared class state, so a second watcher sees them as dupes.
    first = FakeLogWatcher(api, log_batches=[[raw_log()]])
    list(first.fetch_and_parse_logs(None, NOW_MS))
    second = FakeLogWatcher(api, log_batches=[[raw_log()]])
    assert list(second.fetch_and_parse_logs(None, NOW_MS)) == []


def test_poll_queries_open_ended_window_on_first_call(api, fake_time):
    watcher = FakeLogWatcher(api, log_batches=[[raw_log()]])
    logs = list(watcher.poll())
    assert [log.message for log in logs] == ["hello"]
    assert watcher.fetch_calls == [(None, NOW_MS + CLOCK_SKEW_BUFFER_MS)]


def test_poll_records_poll_and_log_times(api, fake_time):
    watcher = FakeLogWatcher(
        api, log_batches=[[raw_log(timestamp="1700000000123456789")]]
    )
    list(watcher.poll())
    assert watcher._last_poll_time_ms == NOW_MS
    assert watcher._last_log_time_ms == 1700000000123


def test_poll_without_logs_leaves_last_log_time_unset(api, fake_time):
    watcher = FakeLogWatcher(api, log_batches=[[]])
    assert list(watcher.poll()) == []
    assert watcher._last_poll_time_ms == NOW_MS
    assert watcher._last_log_time_ms is None


def test_second_poll_starts_from_previous_poll_minus_buffer(api, fake_time):
    watcher = FakeLogWatcher(api, log_batches=[[], []])
    list(watcher.poll())
    fake_time.now_ms = NOW_MS + 5_000
    list(watcher.poll())
    assert watcher.fetch_calls == [
        (None, NOW_MS + CLOCK_SKEW_BUFFER_MS),
        (NOW_MS - CLOCK_SKEW_BUFFER_MS, NOW_MS + 5_000 + CLOCK_SKEW_BUFFER_MS),
    ]


def test_watch_yields_logs_and_runs_lifecycle_hooks(api, fake_time):
    watcher = FakeLogWatcher(api, log_batches=[[raw_log(message="a")]])
    assert [log.message for log in watcher.watch(show_spinner=False)] == ["a"]
    assert watcher.before_polling_calls == 1
    assert watcher.after_polling_calls == 1
    assert watcher.post_poll_calls == 0


def test_watch_polls_until_first_logs_arrive(api, fake_time):
    watcher = FakeLogWatcher(api, log_batches=[[], [], [raw_log(message="a")]])
    assert [log.message for log in watcher.watch(show_spinner=False)] == ["a"]
    assert len(watcher.fetch_calls) == 3
    assert fake_time.sleeps == [POLL_INTERVAL_SEC, POLL_INTERVAL_SEC]


def test_watch_keeps_polling_while_should_poll_again(api, fake_time):
    watcher = FakeLogWatcher(
        api,
        log_batches=[[raw_log(message="a")], [raw_log(message="b")], []],
        poll_again=[True, True, False],
    )
    assert [log.message for log in watcher.watch(show_spinner=False)] == ["a", "b"]
    assert len(watcher.fetch_calls) == 3
    assert watcher.post_poll_calls == 2
    assert fake_time.sleeps == [POLL_INTERVAL_SEC, POLL_INTERVAL_SEC]
    assert watcher.after_polling_calls == 1


def test_watch_shows_spinner_by_default(api, fake_time, monkeypatch):
    console = MagicMock()
    monkeypatch.setattr(base_watcher, "console", console)
    watcher = FakeLogWatcher(api, log_batches=[[raw_log()]])
    list(watcher.watch())
    console.status.assert_called_once_with("Polling logs", spinner="aesthetic")


def test_watch_skips_spinner_when_disabled(api, fake_time, monkeypatch):
    console = MagicMock()
    monkeypatch.setattr(base_watcher, "console", console)
    watcher = FakeLogWatcher(api, log_batches=[[raw_log()]])
    list(watcher.watch(show_spinner=False))
    console.status.assert_not_called()
