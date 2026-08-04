"""Tests for the Loops deployment log watcher."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from truss.cli.logs.loops_deployment_log_watcher import LoopsDeploymentLogWatcher

DEPLOYMENT_ID = "trainer_dep_1"


def _raw_log(message: str, timestamp: str = "1700000000000000000", replica="replica-1"):
    return {"timestamp": timestamp, "message": message, "replica": replica}


@pytest.fixture
def mock_api():
    api = Mock()
    api.get_loops_deployment.return_value = {"status": {"name": "RUNNING"}}
    api.get_loops_deployment_logs.return_value = []
    return api


@pytest.fixture
def watcher(mock_api):
    watcher = LoopsDeploymentLogWatcher(mock_api, DEPLOYMENT_ID)
    # `_log_hashes` is declared on the base class, so give each watcher its own
    # set to keep state from leaking between tests.
    watcher._log_hashes = set()
    return watcher


def test_init_stores_api_and_deployment_id(mock_api):
    watcher = LoopsDeploymentLogWatcher(mock_api, DEPLOYMENT_ID)

    assert watcher.api is mock_api
    assert watcher._loops_deployment_id == DEPLOYMENT_ID
    assert watcher._current_status is None
    mock_api.get_loops_deployment.assert_not_called()


def test_before_polling_fetches_current_status(watcher, mock_api):
    watcher.before_polling()

    assert watcher._current_status == "RUNNING"
    mock_api.get_loops_deployment.assert_called_once_with(DEPLOYMENT_ID)


def test_post_poll_refreshes_current_status(watcher, mock_api):
    watcher.before_polling()
    mock_api.get_loops_deployment.return_value = {"status": {"name": "FAILED"}}

    watcher.post_poll()

    assert watcher._current_status == "FAILED"


def test_after_polling_is_a_noop(watcher, mock_api):
    assert watcher.after_polling() is None
    mock_api.get_loops_deployment.assert_not_called()


def test_fetch_logs_delegates_to_api(watcher, mock_api):
    logs = [_raw_log("hello")]
    mock_api.get_loops_deployment_logs.return_value = logs

    assert watcher.fetch_logs(100, 200) == logs
    mock_api.get_loops_deployment_logs.assert_called_once_with(DEPLOYMENT_ID, 100, 200)


def test_fetch_logs_passes_through_none_bounds(watcher, mock_api):
    assert watcher.fetch_logs(None, None) == []
    mock_api.get_loops_deployment_logs.assert_called_once_with(
        DEPLOYMENT_ID, None, None
    )


@pytest.mark.parametrize("status", ["CREATED", "DEPLOYING", "RUNNING"])
def test_should_poll_again_for_running_statuses(watcher, status):
    watcher._current_status = status

    assert watcher.should_poll_again() is True


@pytest.mark.parametrize(
    "status", ["FAILED", "STOPPED", "UNKNOWN_STATUS", "running", "", None]
)
def test_should_poll_again_false_for_terminal_statuses(watcher, status):
    watcher._current_status = status

    assert watcher.should_poll_again() is False


@pytest.mark.parametrize(
    "deployment, expected",
    [
        ({"status": {"name": "RUNNING"}}, "RUNNING"),
        ({"status": {}}, None),
        ({"status": None}, None),
        ({}, None),
        ({"status": {"name": None}}, None),
    ],
)
def test_get_current_status_handles_partial_responses(
    watcher, mock_api, deployment, expected
):
    mock_api.get_loops_deployment.return_value = deployment

    assert watcher._get_current_status() == expected


def test_get_current_status_propagates_api_errors(watcher, mock_api):
    mock_api.get_loops_deployment.side_effect = RuntimeError("404 not found")

    with pytest.raises(RuntimeError, match="404 not found"):
        watcher._get_current_status()


def test_watch_yields_logs_and_stops_when_deployment_is_not_running(watcher, mock_api):
    mock_api.get_loops_deployment.side_effect = [
        {"status": {"name": "RUNNING"}},  # before_polling
        {"status": {"name": "FAILED"}},  # post_poll
    ]
    mock_api.get_loops_deployment_logs.side_effect = [
        [_raw_log("first")],
        [_raw_log("second", timestamp="1700000001000000000")],
    ]

    with patch("truss.cli.logs.base_watcher.time.sleep"):
        logs = list(watcher.watch(show_spinner=False))

    assert [log.message for log in logs] == ["first", "second"]
    assert watcher._current_status == "FAILED"
    assert mock_api.get_loops_deployment_logs.call_count == 2


def test_watch_polls_until_first_logs_arrive(watcher, mock_api):
    mock_api.get_loops_deployment.side_effect = [
        {"status": {"name": "DEPLOYING"}},  # before_polling
        {"status": {"name": "STOPPED"}},  # post_poll
    ]
    mock_api.get_loops_deployment_logs.side_effect = [[], [], [_raw_log("late")], []]

    with patch("truss.cli.logs.base_watcher.time.sleep") as mock_sleep:
        logs = list(watcher.watch(show_spinner=False))

    assert [log.message for log in logs] == ["late"]
    assert mock_sleep.call_count == 3
    assert mock_api.get_loops_deployment_logs.call_count == 4


def test_watch_deduplicates_repeated_logs(watcher, mock_api):
    mock_api.get_loops_deployment.side_effect = [
        {"status": {"name": "RUNNING"}},  # before_polling
        {"status": {"name": "STOPPED"}},  # post_poll
    ]
    duplicate = _raw_log("same line")
    mock_api.get_loops_deployment_logs.side_effect = [[duplicate], [duplicate]]

    with patch("truss.cli.logs.base_watcher.time.sleep"):
        logs = list(watcher.watch(show_spinner=False))

    assert [log.message for log in logs] == ["same line"]


def test_watch_stops_immediately_when_status_is_terminal(watcher, mock_api):
    mock_api.get_loops_deployment.return_value = {"status": {"name": "FAILED"}}
    mock_api.get_loops_deployment_logs.return_value = [_raw_log("only")]

    with patch("truss.cli.logs.base_watcher.time.sleep") as mock_sleep:
        logs = list(watcher.watch(show_spinner=False))

    assert [log.message for log in logs] == ["only"]
    mock_sleep.assert_not_called()
    mock_api.get_loops_deployment.assert_called_once_with(DEPLOYMENT_ID)
