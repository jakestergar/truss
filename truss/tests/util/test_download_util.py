import os
from pathlib import Path
from unittest import mock

import pytest
import requests
import requests_mock

from truss.base.truss_config import ExternalData
from truss.util import download

TEST_DOWNLOAD_URL = "http://example.com/some-download-url"
OTHER_TEST_DOWNLOAD_URL = "http://example.com/some-other-download-url"


def test_b10cp_path_returns_none_when_env_var_unset():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert download._b10cp_path() is None


def test_b10cp_path_returns_env_var_value():
    with mock.patch.dict(
        os.environ, {download.B10CP_PATH_TRUSS_ENV_VAR_NAME: "/usr/bin/b10cp"}
    ):
        assert download._b10cp_path() == "/usr/bin/b10cp"


def test_download_external_data_none_is_noop(tmp_path):
    data_dir = tmp_path / "data"

    download.download_external_data(external_data=None, data_dir=data_dir)

    assert not data_dir.exists()


def test_download_external_data_empty_items_creates_data_dir(tmp_path):
    data_dir = tmp_path / "data"

    with mock.patch.dict(os.environ, {}, clear=True):
        download.download_external_data(
            external_data=ExternalData([]), data_dir=data_dir
        )

    assert data_dir.is_dir()


@pytest.mark.parametrize("local_data_path", ["foo", "foo/bar/baz"])
def test_download_external_data_using_requests(tmp_path, local_data_path):
    content = b"mocked content"
    external_data = ExternalData(
        [{"local_data_path": local_data_path, "url": TEST_DOWNLOAD_URL}]
    )

    with mock.patch.dict(os.environ, {}, clear=True), requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=content)
        download.download_external_data(external_data=external_data, data_dir=tmp_path)

    assert (tmp_path / local_data_path).read_bytes() == content


def test_download_external_data_downloads_every_item(tmp_path):
    external_data = ExternalData(
        [
            {"local_data_path": "first", "url": TEST_DOWNLOAD_URL},
            {"local_data_path": "nested/second", "url": OTHER_TEST_DOWNLOAD_URL},
        ]
    )

    with mock.patch.dict(os.environ, {}, clear=True), requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"first content")
        m.get(OTHER_TEST_DOWNLOAD_URL, content=b"second content")
        download.download_external_data(external_data=external_data, data_dir=tmp_path)

    assert (tmp_path / "first").read_bytes() == b"first content"
    assert (tmp_path / "nested" / "second").read_bytes() == b"second content"


def test_download_external_data_rejects_path_outside_data_dir(tmp_path):
    outside_path = tmp_path.parent / "outside" / "escaped"
    external_data = ExternalData(
        [{"local_data_path": str(outside_path), "url": TEST_DOWNLOAD_URL}]
    )

    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="outside data directory"):
            download.download_external_data(
                external_data=external_data, data_dir=tmp_path
            )

    assert not outside_path.exists()


def test_download_external_data_using_b10cp(tmp_path):
    external_data = ExternalData(
        [
            {"local_data_path": "first", "url": TEST_DOWNLOAD_URL},
            {"local_data_path": "nested/second", "url": OTHER_TEST_DOWNLOAD_URL},
        ]
    )
    procs = [mock.MagicMock(), mock.MagicMock()]

    with (
        mock.patch.dict(
            os.environ, {download.B10CP_PATH_TRUSS_ENV_VAR_NAME: "/usr/bin/b10cp"}
        ),
        mock.patch.object(download.subprocess, "Popen", side_effect=procs) as popen,
    ):
        download.download_external_data(external_data=external_data, data_dir=tmp_path)

    assert popen.call_args_list == [
        mock.call(
            [
                "/usr/bin/b10cp",
                "-source",
                TEST_DOWNLOAD_URL,
                "-target",
                str((tmp_path / "first").resolve()),
            ]
        ),
        mock.call(
            [
                "/usr/bin/b10cp",
                "-source",
                OTHER_TEST_DOWNLOAD_URL,
                "-target",
                str((tmp_path / "nested" / "second").resolve()),
            ]
        ),
    ]
    for proc in procs:
        proc.wait.assert_called_once_with()
    # Parent directories are created even though b10cp does the downloading.
    assert (tmp_path / "nested").is_dir()


def test_download_external_data_using_b10cp_starts_all_before_waiting(tmp_path):
    """b10cp downloads run concurrently: all are started before any is waited on."""
    events = []

    def make_proc(index):
        proc = mock.MagicMock()
        proc.wait.side_effect = lambda index=index: events.append(("wait", index))
        return proc

    procs = [make_proc(0), make_proc(1)]

    def popen(args):
        events.append(("start", args[2]))
        return procs[len(events) - 1]

    external_data = ExternalData(
        [
            {"local_data_path": "first", "url": TEST_DOWNLOAD_URL},
            {"local_data_path": "second", "url": OTHER_TEST_DOWNLOAD_URL},
        ]
    )

    with (
        mock.patch.dict(
            os.environ, {download.B10CP_PATH_TRUSS_ENV_VAR_NAME: "/usr/bin/b10cp"}
        ),
        mock.patch.object(download.subprocess, "Popen", side_effect=popen),
    ):
        download.download_external_data(external_data=external_data, data_dir=tmp_path)

    assert events == [
        ("start", TEST_DOWNLOAD_URL),
        ("start", OTHER_TEST_DOWNLOAD_URL),
        ("wait", 0),
        ("wait", 1),
    ]


def test_download_from_url_using_requests_writes_streamed_content(tmp_path):
    download_to = tmp_path / "downloaded"

    with requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"streamed content")
        download.download_from_url_using_requests(TEST_DOWNLOAD_URL, download_to)

        assert m.last_request.timeout == download.BLOB_DOWNLOAD_TIMEOUT_SECS

    assert download_to.read_bytes() == b"streamed content"


def test_download_from_url_using_requests_overwrites_existing_file(tmp_path):
    download_to = tmp_path / "downloaded"
    download_to.write_bytes(b"stale content that is longer")

    with requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"new content")
        download.download_from_url_using_requests(TEST_DOWNLOAD_URL, download_to)

    assert download_to.read_bytes() == b"new content"


def test_download_from_url_using_requests_raises_on_error_status(tmp_path):
    download_to = tmp_path / "downloaded"

    with requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, status_code=404)
        with pytest.raises(requests.exceptions.HTTPError):
            download.download_from_url_using_requests(TEST_DOWNLOAD_URL, download_to)

    assert not download_to.exists()


def test_download_from_url_using_requests_propagates_connection_error(tmp_path):
    with requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, exc=requests.exceptions.ConnectTimeout)
        with pytest.raises(requests.exceptions.ConnectTimeout):
            download.download_from_url_using_requests(
                TEST_DOWNLOAD_URL, tmp_path / "downloaded"
            )


def test_download_from_url_using_requests_fails_for_missing_parent_dir(tmp_path):
    download_to: Path = tmp_path / "missing" / "downloaded"

    with requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"content")
        with pytest.raises(FileNotFoundError):
            download.download_from_url_using_requests(TEST_DOWNLOAD_URL, download_to)
