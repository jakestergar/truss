from unittest.mock import Mock, patch

import pytest

from truss.cli.logs.base_watcher import CLOCK_SKEW_BUFFER_MS
from truss.cli.logs.model_log_watcher import MAX_LOOK_BACK_MS, ModelDeploymentLogWatcher
from truss.remote.baseten.api import BasetenApi

NOW_MS = 1_700_000_000_000


def _make_watcher(
    status: str = "ACTIVE", is_development: bool = False
) -> ModelDeploymentLogWatcher:
    api = Mock(spec=BasetenApi)
    api.get_deployment.return_value = {
        "status": status,
        "is_development": is_development,
    }
    api.get_model_deployment_logs.return_value = []
    watcher = ModelDeploymentLogWatcher(api, "model-id", "deployment-id")
    # NB: `_log_hashes` is a class attribute on `LogWatcher`, shadow it per
    # instance so tests don't leak de-duplication state into each other.
    watcher._log_hashes = set()
    return watcher


def _raw_log(timestamp: str, message: str, replica=None) -> dict:
    return {"timestamp": timestamp, "message": message, "replica": replica}


def _stub_polling(
    watcher: ModelDeploymentLogWatcher,
    log_batches: list,
    is_development: bool = False,
    final_status: str = "INACTIVE",
) -> None:
    """Serve one log batch per poll, then report the deployment as stopped."""
    deployment = {"status": "ACTIVE", "is_development": is_development}
    batches = list(log_batches)

    def _fetch(*args):
        if len(batches) <= 1:
            deployment["status"] = final_status
        return batches.pop(0) if batches else []

    watcher.api.get_deployment.side_effect = lambda *args: dict(deployment)
    watcher.api.get_model_deployment_logs.side_effect = _fetch


def test_init_stores_ids_and_api():
    watcher = _make_watcher()
    assert watcher._model_id == "model-id"
    assert watcher._deployment_id == "deployment-id"
    assert watcher._current_status is None


def test_before_polling_sets_current_status():
    watcher = _make_watcher(status="BUILDING")
    watcher.before_polling()
    assert watcher._current_status == "BUILDING"
    watcher.api.get_deployment.assert_called_once_with("model-id", "deployment-id")


def test_fetch_logs_delegates_to_api():
    watcher = _make_watcher()
    watcher.api.get_model_deployment_logs.return_value = [_raw_log("1", "hello")]

    logs = watcher.fetch_logs(100, 200)

    assert logs == [_raw_log("1", "hello")]
    watcher.api.get_model_deployment_logs.assert_called_once_with(
        "model-id", "deployment-id", 100, 200
    )


def test_fetch_logs_passes_none_bounds():
    watcher = _make_watcher()
    watcher.fetch_logs(None, None)
    watcher.api.get_model_deployment_logs.assert_called_once_with(
        "model-id", "deployment-id", None, None
    )


def test_get_start_epoch_ms_non_development_defers_to_base_without_prior_poll():
    watcher = _make_watcher(is_development=False)
    assert watcher.get_start_epoch_ms(NOW_MS) is None


def test_get_start_epoch_ms_non_development_uses_last_poll_with_skew_buffer():
    watcher = _make_watcher(is_development=False)
    watcher._last_poll_time_ms = NOW_MS - 5_000
    # Cursor state is ignored for non-development deployments, which may have
    # multiple replicas.
    watcher._last_log_time_ms = NOW_MS - 1_000

    assert watcher.get_start_epoch_ms(NOW_MS) == NOW_MS - 5_000 - CLOCK_SKEW_BUFFER_MS


def test_get_start_epoch_ms_development_without_cursor_is_none():
    watcher = _make_watcher(is_development=True)
    assert watcher.get_start_epoch_ms(NOW_MS) is None


def test_get_start_epoch_ms_development_uses_recent_cursor():
    watcher = _make_watcher(is_development=True)
    watcher._last_log_time_ms = NOW_MS - 1_000

    assert watcher.get_start_epoch_ms(NOW_MS) == NOW_MS - 1_000


def test_get_start_epoch_ms_development_clamps_stale_cursor_to_look_back():
    watcher = _make_watcher(is_development=True)
    watcher._last_log_time_ms = NOW_MS - (MAX_LOOK_BACK_MS * 2)

    assert watcher.get_start_epoch_ms(NOW_MS) == NOW_MS - MAX_LOOK_BACK_MS


@pytest.mark.parametrize(
    "status",
    ["BUILDING", "DEPLOYING", "LOADING_MODEL", "ACTIVE", "UPDATING", "WAKING_UP"],
)
def test_should_poll_again_true_for_running_states(status):
    watcher = _make_watcher()
    watcher._current_status = status
    assert watcher.should_poll_again() is True


@pytest.mark.parametrize(
    "status", [None, "INACTIVE", "BUILD_FAILED", "DEPLOY_FAILED", "SCALED_TO_ZERO"]
)
def test_should_poll_again_false_for_terminal_states(status):
    watcher = _make_watcher()
    watcher._current_status = status
    assert watcher.should_poll_again() is False


def test_is_development_is_cached():
    watcher = _make_watcher(is_development=True)

    assert watcher._is_development is True
    assert watcher._is_development is True
    watcher.api.get_deployment.assert_called_once_with("model-id", "deployment-id")


def test_post_poll_refreshes_current_status():
    watcher = _make_watcher(status="BUILDING")
    watcher.before_polling()
    watcher.api.get_deployment.return_value = {
        "status": "ACTIVE",
        "is_development": False,
    }

    watcher.post_poll()

    assert watcher._current_status == "ACTIVE"


def test_after_polling_is_a_noop():
    watcher = _make_watcher()
    assert watcher.after_polling() is None


@patch("truss.cli.logs.base_watcher.time.sleep")
def test_watch_yields_logs_until_deployment_stops_running(mock_sleep):
    watcher = _make_watcher()
    _stub_polling(
        watcher,
        [
            [_raw_log("1700000000000000000", "starting", replica="replica-1")],
            [_raw_log("1700000001000000000", "ready")],
        ],
    )

    logs = list(watcher.watch(show_spinner=False))

    assert [log.message for log in logs] == ["starting", "ready"]
    assert logs[0].replica == "replica-1"
    assert watcher._current_status == "INACTIVE"
    mock_sleep.assert_called_once()


@patch("truss.cli.logs.base_watcher.time.sleep")
def test_watch_polls_until_first_logs_arrive(mock_sleep):
    watcher = _make_watcher(is_development=True)
    _stub_polling(
        watcher, [[], [_raw_log("1700000000000000000", "first")]], is_development=True
    )

    logs = list(watcher.watch(show_spinner=False))

    assert [log.message for log in logs] == ["first"]
    # Development deployments use a log-timestamp cursor once one is known.
    assert watcher._last_log_time_ms == 1_700_000_000_000
    # Two polls until the first logs arrive, plus one more before the cached
    # status is refreshed to a non-running state.
    assert watcher.api.get_model_deployment_logs.call_count == 3
    assert watcher._current_status == "INACTIVE"


@patch("truss.cli.logs.base_watcher.time.sleep")
def test_watch_deduplicates_repeated_logs(mock_sleep):
    watcher = _make_watcher()
    duplicate = _raw_log("1700000000000000000", "same")
    _stub_polling(watcher, [[duplicate], [duplicate]])

    logs = list(watcher.watch(show_spinner=False))

    assert [log.message for log in logs] == ["same"]
