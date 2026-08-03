from unittest.mock import Mock, patch

import pytest

from truss.base.truss_config import ExternalData, ExternalDataItem
from truss.util.download import (
    _b10cp_path,
    _download_external_data_using_b10cp,
    _download_from_url_using_b10cp,
    download_external_data,
    download_from_url_using_requests,
)


def external_data(*paths):
    return ExternalData(
        [
            ExternalDataItem(url=f"https://example/{path}", local_data_path=path)
            for path in paths
        ]
    )


def test_download_external_data_none_is_noop(tmp_path):
    with patch("truss.util.download._b10cp_path") as b10cp:
        download_external_data(None, tmp_path / "data")
    b10cp.assert_not_called()
    assert not (tmp_path / "data").exists()


def test_download_external_data_rejects_path_escape(tmp_path):
    data = external_data("/tmp/outside")
    with pytest.raises(ValueError, match="outside data directory"):
        download_external_data(data, tmp_path / "data")


def test_download_external_data_uses_b10cp_and_creates_parents(tmp_path):
    data = external_data("nested/file.bin", "second.bin")
    with (
        patch("truss.util.download._b10cp_path", return_value="/bin/b10cp"),
        patch("truss.util.download._download_external_data_using_b10cp") as download,
    ):
        download_external_data(data, tmp_path / "data")
    download.assert_called_once_with("/bin/b10cp", tmp_path / "data", data)
    assert (tmp_path / "data" / "nested").is_dir()


def test_download_external_data_uses_requests_when_b10cp_absent(tmp_path):
    data = external_data("nested/file.bin")
    with (
        patch("truss.util.download._b10cp_path", return_value=None),
        patch("truss.util.download._download_external_data_using_requests") as download,
    ):
        download_external_data(data, tmp_path / "data")
    download.assert_called_once_with(tmp_path / "data", data)


def test_b10cp_download_starts_and_waits_for_each_item(tmp_path):
    data = external_data("a", "nested/b")
    first, second = Mock(), Mock()
    with patch(
        "truss.util.download._download_from_url_using_b10cp",
        side_effect=[first, second],
    ) as download:
        _download_external_data_using_b10cp("/bin/b10cp", tmp_path, data)
    assert download.call_count == 2
    first.wait.assert_called_once_with()
    second.wait.assert_called_once_with()


def test_download_from_url_using_b10cp_command(tmp_path):
    with patch("truss.util.download.subprocess.Popen") as popen:
        result = _download_from_url_using_b10cp(
            "/bin/b10cp", "https://example/file?q=a b", tmp_path / "file"
        )
    assert result is popen.return_value
    popen.assert_called_once_with(
        [
            "/bin/b10cp",
            "-source",
            "https://example/file?q=a b",
            "-target",
            str(tmp_path / "file"),
        ]
    )


def test_download_from_url_using_requests_streams_to_file(tmp_path):
    raw = Mock()
    raw.read.side_effect = [b"hello", b""]
    resp = Mock(raw=raw)
    with patch("truss.util.download.requests.get", return_value=resp) as get:
        download_from_url_using_requests("https://example/file", tmp_path / "file")
    get.assert_called_once_with(
        "https://example/file", allow_redirects=True, stream=True, timeout=600
    )
    resp.raise_for_status.assert_called_once_with()
    assert (tmp_path / "file").read_bytes() == b"hello"


def test_b10cp_path_reads_environment(monkeypatch):
    monkeypatch.setenv("B10CP_PATH_TRUSS", "/custom/b10cp")
    assert _b10cp_path() == "/custom/b10cp"
