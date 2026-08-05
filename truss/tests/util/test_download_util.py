import os
from unittest import mock

import pytest
import requests
import requests_mock

from truss.base.truss_config import ExternalData
from truss.util import download

TEST_DOWNLOAD_URL = "http://example.com/some-download-url"


def _external_data(local_data_path: str = "foo") -> ExternalData:
    return ExternalData(
        [{"local_data_path": local_data_path, "url": TEST_DOWNLOAD_URL}]
    )


def test_download_external_data_without_data_is_noop(tmp_path):
    data_dir = tmp_path / "data"

    download.download_external_data(external_data=None, data_dir=data_dir)

    assert not data_dir.exists()


def test_download_external_data_rejects_path_outside_data_dir(tmp_path):
    with pytest.raises(ValueError, match="cannot point to outside data directory"):
        download.download_external_data(
            external_data=_external_data("/etc/passwd"), data_dir=tmp_path
        )


def test_download_external_data_uses_b10cp_when_available(tmp_path):
    proc = mock.MagicMock()
    with (
        mock.patch.dict(
            os.environ, {download.B10CP_PATH_TRUSS_ENV_VAR_NAME: "/bin/b10cp"}
        ),
        mock.patch.object(download.subprocess, "Popen", return_value=proc) as popen,
    ):
        download.download_external_data(
            external_data=_external_data("nested/foo"), data_dir=tmp_path
        )

    popen.assert_called_once_with(
        [
            "/bin/b10cp",
            "-source",
            TEST_DOWNLOAD_URL,
            "-target",
            str((tmp_path / "nested" / "foo").resolve()),
        ]
    )
    proc.wait.assert_called_once()
    assert (tmp_path / "nested").is_dir()


def test_download_from_url_using_requests_streams_to_file(tmp_path):
    download_to = tmp_path / "downloaded"
    with requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"mocked content")
        download.download_from_url_using_requests(TEST_DOWNLOAD_URL, download_to)

    assert download_to.read_bytes() == b"mocked content"


def test_download_from_url_using_requests_raises_for_error_status(tmp_path):
    with requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, status_code=404)
        with pytest.raises(requests.exceptions.HTTPError):
            download.download_from_url_using_requests(
                TEST_DOWNLOAD_URL, tmp_path / "downloaded"
            )
