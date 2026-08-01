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


class _CheckProbaOk(_WithProba):
    def _check_proba(self):
        return True


class _CheckProbaRaises(_WithProba):
    def _check_proba(self):
        raise AttributeError("not available")


@pytest.mark.parametrize(
    "model, expected",
    [
        (_NoProba(), False),
        (_WithProba(), True),
        (_CheckProbaOk(), True),
        (_CheckProbaRaises(), False),
    ],
)
def test_model_supports_predict_proba(model, expected):
    assert util.model_supports_predict_proba(model) is expected


def test_cpu_count_takes_minimum_of_os_and_affinity():
    with (
        mock.patch("os.cpu_count", return_value=16),
        mock.patch.object(psutil.Process, "cpu_affinity", return_value=[0, 1, 2, 3]),
        mock.patch("sys.platform", "darwin"),
    ):
        assert util.cpu_count() == 4


def test_cpu_count_ignores_unavailable_affinity():
    with (
        mock.patch("os.cpu_count", return_value=2),
        mock.patch.object(
            psutil.Process, "cpu_affinity", side_effect=AttributeError("unsupported")
        ),
        mock.patch("sys.platform", "darwin"),
    ):
        assert util.cpu_count() == 2


def test_cpu_count_ignores_empty_affinity():
    with (
        mock.patch("os.cpu_count", return_value=2),
        mock.patch.object(psutil.Process, "cpu_affinity", return_value=[]),
        mock.patch("sys.platform", "darwin"),
    ):
        assert util.cpu_count() == 2


def _cgroup_reader(quota: str, period: str):
    def _open(path, *args, **kwargs):
        contents = {
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us": quota,
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us": period,
        }[path]
        return mock.mock_open(read_data=contents)(path, *args, **kwargs)

    return _open


def test_cpu_count_applies_cgroups_limit_on_linux():
    with (
        mock.patch("os.cpu_count", return_value=16),
        mock.patch.object(psutil.Process, "cpu_affinity", return_value=list(range(16))),
        mock.patch("sys.platform", "linux"),
        mock.patch("builtins.open", _cgroup_reader("200000", "100000")),
    ):
        assert util.cpu_count() == 2


def test_cpu_count_ignores_non_positive_cgroups_quota():
    with (
        mock.patch("os.cpu_count", return_value=8),
        mock.patch.object(psutil.Process, "cpu_affinity", return_value=list(range(8))),
        mock.patch("sys.platform", "linux"),
        mock.patch("builtins.open", _cgroup_reader("-1", "100000")),
    ):
        assert util.cpu_count() == 8


def test_cpu_count_ignores_missing_cgroups_files():
    with (
        mock.patch("os.cpu_count", return_value=8),
        mock.patch.object(psutil.Process, "cpu_affinity", return_value=list(range(8))),
        mock.patch("sys.platform", "linux"),
        mock.patch("builtins.open", side_effect=FileNotFoundError),
    ):
        assert util.cpu_count() == 8


def test_all_processes_dead_with_no_processes():
    assert util.all_processes_dead([]) is True


def test_all_processes_dead_reports_alive_process():
    alive = mock.Mock(spec=multiprocessing.Process)
    alive.is_alive.return_value = True
    dead = mock.Mock(spec=multiprocessing.Process)
    dead.is_alive.return_value = False

    assert util.all_processes_dead([dead, alive]) is False
    assert util.all_processes_dead([dead, dead]) is True


def test_kill_child_processes_terminates_children():
    child = mock.Mock()
    parent = mock.Mock()
    parent.children.return_value = [child]
    with (
        mock.patch.object(psutil, "Process", return_value=parent) as process,
        mock.patch.object(psutil, "wait_procs", return_value=([child], [])) as wait,
    ):
        util.kill_child_processes(123)

    process.assert_called_once_with(123)
    parent.children.assert_called_once_with(recursive=True)
    wait.assert_called_once_with(
        [child], timeout=util.CHILD_PROCESS_WAIT_TIMEOUT_SECONDS
    )
    child.terminate.assert_called_once_with()
    child.kill.assert_not_called()


def test_kill_child_processes_kills_survivors():
    child = mock.Mock()
    parent = mock.Mock()
    parent.children.return_value = [child]
    with (
        mock.patch.object(psutil, "Process", return_value=parent),
        mock.patch.object(psutil, "wait_procs", return_value=([], [child])),
    ):
        util.kill_child_processes(123, timeout_seconds=0.1)

    child.terminate.assert_called_once_with()
    child.kill.assert_called_once_with()


def test_kill_child_processes_ignores_missing_parent():
    with mock.patch.object(psutil, "Process", side_effect=psutil.NoSuchProcess(1)):
        util.kill_child_processes(1)
