from unittest.mock import Mock, patch

import pytest

from truss.cli.logs.base_watcher import CLOCK_SKEW_BUFFER_MS, LogWatcher
from truss.cli.logs.loops_deployment_log_watcher import LoopsDeploymentLogWatcher
from truss.cli.logs.model_log_watcher import ModelDeploymentLogWatcher
from truss.cli.logs.utils import ParsedLog


class FakeWatcher(LogWatcher):
    def __init__(self, logs=None, statuses=None):
        super().__init__(Mock())
        self.logs = logs or []
        self.statuses = iter(statuses or [])
        self.events = []

    def fetch_logs(self, start_epoch_millis, end_epoch_millis):
        self.events.append(("fetch", start_epoch_millis, end_epoch_millis))
        return self.logs

    def before_polling(self):
        self.events.append("before")

    def after_polling(self):
        self.events.append("after")

    def should_poll_again(self):
        return bool(next(self.statuses, False))

    def post_poll(self):
        self.events.append("post")


@pytest.fixture(autouse=True)
def clear_log_watcher_state():
    LogWatcher._log_hashes = set()
    LogWatcher._last_poll_time_ms = None
    LogWatcher._last_log_time_ms = None


def test_fetch_and_parse_logs_deduplicates_logs():
    raw = [{"timestamp": "1000000000", "message": " hello ", "replica": "r1"}]
    watcher = FakeWatcher(raw)
    assert list(watcher.fetch_and_parse_logs(None, 5)) == [
        ParsedLog(timestamp="1000000000", message=" hello ", replica="r1")
    ]
    assert list(watcher.fetch_and_parse_logs(None, 5)) == []


def test_poll_tracks_cursor_and_last_log_time():
    watcher = FakeWatcher(
        [{"timestamp": "2000000000", "message": "x", "replica": None}]
    )
    with patch("truss.cli.logs.base_watcher.time.time", return_value=10):
        logs = list(watcher.poll())
    assert len(logs) == 1
    assert watcher._last_log_time_ms == 2000
    assert watcher._last_poll_time_ms == 10000
    assert watcher.events == [("fetch", None, 10000 + CLOCK_SKEW_BUFFER_MS)]


def test_get_start_epoch_uses_clock_skew_buffer():
    watcher = FakeWatcher()
    watcher._last_poll_time_ms = 123456
    assert watcher.get_start_epoch_ms(999) == 123456 - CLOCK_SKEW_BUFFER_MS


def test_watch_runs_hooks_and_repolls_until_status_stops():
    logs = [{"timestamp": "1000000000", "message": "x", "replica": None}]
    watcher = FakeWatcher(logs, statuses=[True, False])
    with patch("truss.cli.logs.base_watcher.time.sleep"):
        assert len(list(watcher.watch(show_spinner=False))) == 1
    assert watcher.events[0] == "before"
    assert "post" in watcher.events
    assert watcher.events[-1] == "after"


def test_model_watcher_calls_model_api_and_development_cursor():
    api = Mock()
    api.get_deployment.return_value = {"status": "ACTIVE", "is_development": True}
    api.get_model_deployment_logs.return_value = []
    watcher = ModelDeploymentLogWatcher(api, "model", "deployment")
    watcher.before_polling()
    assert watcher.should_poll_again()
    assert watcher.fetch_logs(1, 2) == []
    api.get_model_deployment_logs.assert_called_once_with("model", "deployment", 1, 2)
    watcher._last_log_time_ms = 5000
    assert watcher.get_start_epoch_ms(100000) == 5000
    watcher.post_poll()
    assert watcher._current_status == "ACTIVE"


def test_model_watcher_non_development_uses_base_cursor():
    api = Mock()
    api.get_deployment.return_value = {"status": "FAILED", "is_development": False}
    watcher = ModelDeploymentLogWatcher(api, "model", "deployment")
    watcher._last_poll_time_ms = 100000
    assert watcher.get_start_epoch_ms(1) == 100000 - CLOCK_SKEW_BUFFER_MS
    watcher.before_polling()
    assert not watcher.should_poll_again()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": {"name": "RUNNING"}}, "RUNNING"),
        ({"status": {}}, None),
        ({}, None),
        ({"status": None}, None),
    ],
)
def test_loops_watcher_status_and_fetch(payload, expected):
    api = Mock()
    api.get_loops_deployment.return_value = payload
    api.get_loops_deployment_logs.return_value = ["raw"]
    watcher = LoopsDeploymentLogWatcher(api, "loops-id")
    watcher.before_polling()
    assert watcher._current_status == expected
    assert watcher.fetch_logs(1, 2) == ["raw"]
    api.get_loops_deployment_logs.assert_called_once_with("loops-id", 1, 2)
    assert watcher.should_poll_again() is (expected == "RUNNING")


def test_loops_watcher_post_poll_refreshes_status():
    api = Mock()
    api.get_loops_deployment.side_effect = [
        {"status": {"name": "DEPLOYING"}},
        {"status": {"name": "STOPPED"}},
    ]
    watcher = LoopsDeploymentLogWatcher(api, "loops-id")
    watcher.before_polling()
    assert watcher.should_poll_again()
    watcher.post_poll()
    assert not watcher.should_poll_again()
