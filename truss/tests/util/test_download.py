from unittest import mock

import pytest

from truss.base.truss_config import ExternalData, ExternalDataItem
from truss.util.download import download_external_data, download_from_url_using_requests


def test_download_external_data_noop_for_none(tmp_path):
    download_external_data(None, tmp_path)
    assert not any(tmp_path.iterdir())


def test_download_external_data_with_requests(tmp_path):
    item = ExternalDataItem(
        url="http://example.com/data.bin", local_data_path="data.bin"
    )
    external_data = ExternalData([item])
    content = b"hello world"
    resp = mock.MagicMock()
    resp.raw = mock.MagicMock()
    resp.raw.read.side_effect = [content, b""]

    with (
        mock.patch("truss.util.download.requests.get", return_value=resp) as mock_get,
        mock.patch("truss.util.download._b10cp_path", return_value=None),
    ):
        download_external_data(external_data, tmp_path)

    mock_get.assert_called_once_with(
        "http://example.com/data.bin", allow_redirects=True, stream=True, timeout=600
    )
    assert (tmp_path / "data.bin").read_bytes() == content


def test_download_external_data_prevents_path_escape(tmp_path):
    item = ExternalDataItem(
        url="http://example.com/data.bin", local_data_path="/outside.bin"
    )
    external_data = ExternalData([item])
    with pytest.raises(ValueError, match="outside data directory"):
        download_external_data(external_data, tmp_path)


def test_download_external_data_uses_b10cp(tmp_path):
    item = ExternalDataItem(
        url="http://example.com/data.bin", local_data_path="data.bin"
    )
    external_data = ExternalData([item])
    proc = mock.MagicMock()

    with (
        mock.patch("truss.util.download._b10cp_path", return_value="/usr/bin/b10cp"),
        mock.patch(
            "truss.util.download._download_from_url_using_b10cp", return_value=proc
        ) as mock_download,
    ):
        download_external_data(external_data, tmp_path)

    mock_download.assert_called_once()
    proc.wait.assert_called_once()


def test_download_external_data_b10cp_url_encoding(tmp_path):
    download_to = tmp_path / "data.bin"
    with mock.patch("truss.util.download.subprocess.Popen") as mock_popen:
        from truss.util.download import _download_from_url_using_b10cp

        _download_from_url_using_b10cp(
            "/bin/b10cp", "http://example.com/a?b=1", download_to
        )
        mock_popen.assert_called_once_with(
            [
                "/bin/b10cp",
                "-source",
                "http://example.com/a?b=1",
                "-target",
                str(download_to),
            ]
        )


def test_download_from_url_using_requests(tmp_path):
    url = "http://example.com/blob"
    download_to = tmp_path / "blob"
    content = b"payload"
    resp = mock.MagicMock()
    resp.raw = mock.MagicMock()
    resp.raw.read.side_effect = [content, b""]

    with mock.patch("truss.util.download.requests.get", return_value=resp) as mock_get:
        download_from_url_using_requests(url, download_to)

    mock_get.assert_called_once_with(
        url, allow_redirects=True, stream=True, timeout=600
    )
    resp.raise_for_status.assert_called_once()
    assert download_to.read_bytes() == content
