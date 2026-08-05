import multiprocessing
from unittest import mock

import psutil
import pytest

from truss.templates.shared import util


class _NoProba:
    pass


class _WithProba:
    def predict_proba(self, x):
        return x


class _WithCheckProba(_WithProba):
    def _check_proba(self):
        return True


class _WithFailingCheckProba(_WithProba):
    def _check_proba(self):
        raise AttributeError("predict_proba is not available for probability=False")


@pytest.mark.parametrize(
    "model_class, expected",
    [
        (_NoProba, False),
        (_WithProba, True),
        (_WithCheckProba, True),
        (_WithFailingCheckProba, False),
    ],
)
def test_model_supports_predict_proba(model_class, expected):
    assert util.model_supports_predict_proba(model_class()) is expected


def _cgroup_files(quota: str, period: str):
    quota_file = mock.mock_open(read_data=quota).return_value
    period_file = mock.mock_open(read_data=period).return_value
    return mock.Mock(side_effect=[quota_file, period_file])


def test_cpu_count_uses_system_count_when_no_limits_apply():
    with (
        mock.patch.object(util.os, "cpu_count", return_value=8),
        mock.patch.object(util.psutil, "Process") as process,
        mock.patch.object(util.sys, "platform", "darwin"),
    ):
        process.return_value.cpu_affinity.return_value = list(range(8))
        assert util.cpu_count() == 8


def test_cpu_count_limited_by_cpu_affinity():
    with (
        mock.patch.object(util.os, "cpu_count", return_value=8),
        mock.patch.object(util.psutil, "Process") as process,
        mock.patch.object(util.sys, "platform", "darwin"),
    ):
        process.return_value.cpu_affinity.return_value = [0, 1]
        assert util.cpu_count() == 2


def test_cpu_count_ignores_unavailable_cpu_affinity():
    with (
        mock.patch.object(util.os, "cpu_count", return_value=4),
        mock.patch.object(
            util.psutil, "Process", side_effect=NotImplementedError("no affinity")
        ),
        mock.patch.object(util.sys, "platform", "darwin"),
    ):
        assert util.cpu_count() == 4


def test_cpu_count_limited_by_cgroups_quota():
    with (
        mock.patch.object(util.os, "cpu_count", return_value=8),
        mock.patch.object(util.psutil, "Process") as process,
        mock.patch.object(util.sys, "platform", "linux"),
        mock.patch("builtins.open", _cgroup_files("200000", "100000")),
    ):
        process.return_value.cpu_affinity.return_value = list(range(8))
        assert util.cpu_count() == 2


def test_cpu_count_ignores_unlimited_cgroups_quota():
    with (
        mock.patch.object(util.os, "cpu_count", return_value=8),
        mock.patch.object(util.psutil, "Process") as process,
        mock.patch.object(util.sys, "platform", "linux"),
        mock.patch("builtins.open", _cgroup_files("-1", "100000")),
    ):
        process.return_value.cpu_affinity.return_value = list(range(8))
        assert util.cpu_count() == 8


def test_cpu_count_ignores_missing_cgroups_files():
    with (
        mock.patch.object(util.os, "cpu_count", return_value=8),
        mock.patch.object(util.psutil, "Process") as process,
        mock.patch.object(util.sys, "platform", "linux"),
        mock.patch("builtins.open", side_effect=FileNotFoundError),
    ):
        process.return_value.cpu_affinity.return_value = list(range(8))
        assert util.cpu_count() == 8


def test_cpu_count_on_real_system_is_positive():
    assert util.cpu_count() >= 1


@pytest.mark.parametrize(
    "alive_flags, expected",
    [([], True), ([False, False], True), ([False, True], False)],
)
def test_all_processes_dead(alive_flags, expected):
    procs = []
    for is_alive in alive_flags:
        proc = mock.create_autospec(multiprocessing.Process, instance=True)
        proc.is_alive.return_value = is_alive
        procs.append(proc)

    assert util.all_processes_dead(procs) is expected


def test_kill_child_processes_terminates_then_kills_survivors():
    terminated = mock.MagicMock()
    survivor = mock.MagicMock()
    parent = mock.MagicMock()
    parent.children.return_value = [terminated, survivor]

    with (
        mock.patch.object(util.psutil, "Process", return_value=parent),
        mock.patch.object(
            util.psutil, "wait_procs", return_value=([terminated], [survivor])
        ) as wait_procs,
    ):
        util.kill_child_processes(123, timeout_seconds=5)

    parent.children.assert_called_once_with(recursive=True)
    terminated.terminate.assert_called_once()
    survivor.terminate.assert_called_once()
    wait_procs.assert_called_once_with([terminated, survivor], timeout=5)
    survivor.kill.assert_called_once()
    terminated.kill.assert_not_called()


def test_kill_child_processes_without_parent_process():
    with (
        mock.patch.object(util.psutil, "Process", side_effect=psutil.NoSuchProcess(1)),
        mock.patch.object(util.psutil, "wait_procs") as wait_procs,
    ):
        util.kill_child_processes(1)

    wait_procs.assert_not_called()
