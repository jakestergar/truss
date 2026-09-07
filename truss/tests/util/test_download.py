import os
from pathlib import Path
from unittest.mock import patch

import pytest

from truss.base.truss_config import ExternalData
from truss.util import download


class FakeProc:
    def __init__(self, tracker: "FakeB10cp", returncode: int):
        self._tracker = tracker
        self._returncode = returncode

    def wait(self) -> int:
        self._tracker.in_flight -= 1
        return self._returncode

    def kill(self) -> None:
        self._tracker.in_flight -= 1
        self._tracker.killed += 1


class FakeB10cp:
    def __init__(self, returncodes=None):
        self.returncodes = returncodes or {}
        self.in_flight = 0
        self.peak_in_flight = 0
        self.killed = 0
        self.calls: list = []

    def __call__(self, b10cp_path: str, url: str, download_to: Path) -> FakeProc:
        self.calls.append((url, download_to))
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        return FakeProc(self, self.returncodes.get(url, 0))


def _external_data(count: int) -> ExternalData:
    return ExternalData(
        [
            {"local_data_path": f"file-{i}", "url": f"http://example.com/{i}"}
            for i in range(count)
        ]
    )


def test_b10cp_concurrency_is_bounded_by_default(tmp_path):
    fake = FakeB10cp()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(download, "_download_from_url_using_b10cp", fake),
    ):
        download._download_external_data_using_b10cp(
            "b10cp", tmp_path, _external_data(25)
        )

    assert fake.peak_in_flight == download.B10CP_DEFAULT_MAX_CONCURRENCY
    assert len(fake.calls) == 25
    assert fake.in_flight == 0


def test_b10cp_concurrency_respects_env_var(tmp_path):
    fake = FakeB10cp()
    with (
        patch.dict(
            os.environ, {download.B10CP_MAX_CONCURRENCY_ENV_VAR_NAME: "3"}, clear=True
        ),
        patch.object(download, "_download_from_url_using_b10cp", fake),
    ):
        download._download_external_data_using_b10cp(
            "b10cp", tmp_path, _external_data(10)
        )

    assert fake.peak_in_flight == 3


def test_b10cp_non_zero_exit_raises(tmp_path):
    failing_url = "http://example.com/1"
    fake = FakeB10cp(returncodes={failing_url: 3})
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(download, "_download_from_url_using_b10cp", fake),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            download._download_external_data_using_b10cp(
                "b10cp", tmp_path, _external_data(3)
            )

    message = str(exc_info.value)
    assert failing_url in message
    assert str((tmp_path / "file-1").resolve()) in message
    assert "3" in message
    assert fake.in_flight == 0


def test_b10cp_invalid_concurrency_raises(tmp_path):
    with patch.dict(
        os.environ, {download.B10CP_MAX_CONCURRENCY_ENV_VAR_NAME: "0"}, clear=True
    ):
        with pytest.raises(ValueError):
            download._download_external_data_using_b10cp(
                "b10cp", tmp_path, _external_data(1)
            )
