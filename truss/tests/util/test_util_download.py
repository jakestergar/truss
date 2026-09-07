import subprocess
from pathlib import Path
from unittest import mock

import pytest
import requests
import requests_mock

from truss.base.truss_config import ExternalData, ExternalDataItem
from truss.util import download
from truss.util.download import (
    BLOB_DOWNLOAD_TIMEOUT_SECS,
    download_external_data,
    download_from_url_using_requests,
)


def _external_data(*pairs: tuple[str, str]) -> ExternalData:
    return ExternalData(
        [
            ExternalDataItem(url=url, local_data_path=local_data_path)
            for url, local_data_path in pairs
        ]
    )


@pytest.fixture
def no_b10cp(monkeypatch):
    monkeypatch.delenv("B10CP_PATH_TRUSS", raising=False)


@pytest.fixture
def fake_popen():
    with mock.patch.object(subprocess, "Popen") as popen:
        popen.return_value.wait.return_value = 0
        yield popen


def test_download_external_data_none_is_noop(tmp_path, no_b10cp):
    data_dir = tmp_path / "data"
    download_external_data(None, data_dir)
    assert not data_dir.exists()


def test_download_external_data_empty_items_creates_data_dir(tmp_path, no_b10cp):
    data_dir = tmp_path / "data"
    download_external_data(_external_data(), data_dir)
    assert data_dir.is_dir()


def test_download_external_data_existing_data_dir_is_ok(tmp_path, no_b10cp):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "keep.txt").write_text("keep")
    download_external_data(_external_data(), data_dir)
    assert (data_dir / "keep.txt").read_text() == "keep"


def test_download_external_data_data_dir_missing_parent_raises(tmp_path, no_b10cp):
    # `data_dir.mkdir(exist_ok=True)` is called without `parents=True`, so a
    # data dir nested under a non-existent directory fails.
    data_dir = tmp_path / "missing" / "data"
    with pytest.raises(FileNotFoundError):
        download_external_data(_external_data(), data_dir)


def test_download_external_data_absolute_local_data_path_rejected(tmp_path, no_b10cp):
    data_dir = tmp_path / "data"
    with pytest.raises(ValueError, match="outside data directory"):
        download_external_data(
            _external_data(("https://example.com/f", "/etc/passwd")), data_dir
        )


def test_download_external_data_creates_nested_parent_dirs(tmp_path, no_b10cp):
    data_dir = tmp_path / "data"
    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", content=b"payload")
        download_external_data(
            _external_data(("https://example.com/f", "nested/deep/f.bin")), data_dir
        )
    assert (data_dir / "nested" / "deep" / "f.bin").read_bytes() == b"payload"


def test_download_external_data_normalizes_dot_segments(tmp_path, no_b10cp):
    data_dir = tmp_path / "data"
    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", content=b"payload")
        download_external_data(
            _external_data(("https://example.com/f", "sub/./x.bin")), data_dir
        )
    assert (data_dir / "sub" / "x.bin").read_bytes() == b"payload"


@pytest.mark.parametrize("local_data_path", ["../evil.txt", "a/../../evil.txt"])
def test_download_external_data_relative_traversal_escapes_data_dir_bug(
    tmp_path, no_b10cp, local_data_path
):
    # BUG: the traversal guard `if data_dir not in path.parents` is purely lexical
    # and is evaluated on the *unresolved* path, so `data_dir/"../evil.txt"` has
    # `data_dir` among its parents and passes the check. The actual write later
    # calls `.resolve()`, which collapses the `..` and writes outside `data_dir`.
    # This test asserts the current (wrong) behavior so CI stays green.
    data_dir = tmp_path / "data"
    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", content=b"owned")
        download_external_data(
            _external_data(("https://example.com/f", local_data_path)), data_dir
        )

    escaped = (data_dir / local_data_path).resolve()
    assert data_dir not in escaped.parents  # the file landed outside the data dir
    assert escaped == tmp_path / "evil.txt"
    assert escaped.read_bytes() == b"owned"


def test_download_external_data_symlinked_path_escapes_data_dir_bug(tmp_path, no_b10cp):
    # Same bug via a symlink inside the data dir: the lexical guard cannot see
    # that `data_dir/link` resolves outside `data_dir`.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data_dir / "link").symlink_to(outside, target_is_directory=True)

    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", content=b"owned")
        download_external_data(
            _external_data(("https://example.com/f", "link/evil.txt")), data_dir
        )

    assert (outside / "evil.txt").read_bytes() == b"owned"


def test_b10cp_path_used_when_env_var_set(tmp_path, monkeypatch, fake_popen):
    monkeypatch.setenv("B10CP_PATH_TRUSS", "/usr/bin/b10cp")
    data_dir = tmp_path / "data"
    download_external_data(
        _external_data(
            ("https://example.com/a", "a.bin"), ("https://example.com/b", "sub/b.bin")
        ),
        data_dir,
    )

    assert fake_popen.call_count == 2
    assert [call.args[0] for call in fake_popen.call_args_list] == [
        [
            "/usr/bin/b10cp",
            "-source",
            "https://example.com/a",
            "-target",
            str((data_dir / "a.bin").resolve()),
        ],
        [
            "/usr/bin/b10cp",
            "-source",
            "https://example.com/b",
            "-target",
            str((data_dir / "sub" / "b.bin").resolve()),
        ],
    ]
    # every spawned process is waited on
    assert fake_popen.return_value.wait.call_count == 2
    # parent dirs are created before the downloads start
    assert (data_dir / "sub").is_dir()


def test_b10cp_not_used_when_env_var_unset(tmp_path, no_b10cp, fake_popen):
    with requests_mock.Mocker() as m:
        m.get("https://example.com/a", content=b"payload")
        download_external_data(
            _external_data(("https://example.com/a", "a.bin")), tmp_path / "data"
        )
    fake_popen.assert_not_called()


def test_b10cp_processes_started_before_any_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("B10CP_PATH_TRUSS", "b10cp")
    events = []

    def make_proc(idx):
        proc = mock.Mock()
        proc.wait.side_effect = lambda: events.append(f"wait{idx}") or 0
        return proc

    procs = [make_proc(0), make_proc(1)]

    def popen(argv):
        events.append(f"popen{len(events)}")
        return procs[len([e for e in events if e.startswith("popen")]) - 1]

    with mock.patch.object(subprocess, "Popen", side_effect=popen):
        download_external_data(
            _external_data(
                ("https://example.com/a", "a.bin"), ("https://example.com/b", "b.bin")
            ),
            tmp_path / "data",
        )

    assert events == ["popen0", "popen1", "wait0", "wait1"]


def test_b10cp_nonzero_exit_is_ignored_bug(tmp_path, monkeypatch, fake_popen):
    # BUG: `proc.wait()`'s return code is never inspected, so a failed b10cp
    # download is indistinguishable from a successful one -- the caller proceeds
    # with a missing (or truncated) weights file. Asserting current behavior.
    monkeypatch.setenv("B10CP_PATH_TRUSS", "b10cp")
    fake_popen.return_value.wait.return_value = 1
    data_dir = tmp_path / "data"

    download_external_data(_external_data(("https://example.com/a", "a.bin")), data_dir)

    assert not (data_dir / "a.bin").exists()  # no error raised, no file downloaded


def test_download_from_url_using_requests_streams_content(tmp_path):
    target = tmp_path / "out.bin"
    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", content=b"x" * 4096)
        download_from_url_using_requests("https://example.com/f", target)
    assert target.read_bytes() == b"x" * 4096


def test_download_from_url_using_requests_passes_request_options(tmp_path):
    target = tmp_path / "out.bin"
    with mock.patch.object(download.requests, "get") as get:
        get.return_value.raw = mock.MagicMock()
        get.return_value.raw.read.side_effect = [b"data", b""]
        download_from_url_using_requests("https://example.com/f", target)

    get.assert_called_once_with(
        "https://example.com/f",
        allow_redirects=True,
        stream=True,
        timeout=BLOB_DOWNLOAD_TIMEOUT_SECS,
    )
    assert BLOB_DOWNLOAD_TIMEOUT_SECS == 600


@pytest.mark.parametrize("status_code", [404, 500])
def test_download_from_url_using_requests_raises_for_status(tmp_path, status_code):
    target = tmp_path / "out.bin"
    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", status_code=status_code)
        with pytest.raises(requests.HTTPError):
            download_from_url_using_requests("https://example.com/f", target)
    assert not target.exists()


def test_download_from_url_using_requests_propagates_timeout(tmp_path):
    target = tmp_path / "out.bin"
    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", exc=requests.exceptions.ConnectTimeout)
        with pytest.raises(requests.exceptions.ConnectTimeout):
            download_from_url_using_requests("https://example.com/f", target)
    assert not target.exists()


def test_download_from_url_using_requests_follows_redirects(tmp_path):
    target = tmp_path / "out.bin"
    with requests_mock.Mocker() as m:
        m.get(
            "https://example.com/f",
            status_code=302,
            headers={"Location": "https://cdn.example.com/f"},
        )
        m.get("https://cdn.example.com/f", content=b"redirected")
        download_from_url_using_requests("https://example.com/f", target)
    assert target.read_bytes() == b"redirected"


def test_download_from_url_using_requests_requires_existing_parent(tmp_path):
    # `download_from_url_using_requests` itself does not create parent dirs;
    # `download_external_data` is responsible for that.
    target: Path = tmp_path / "missing" / "out.bin"
    with requests_mock.Mocker() as m:
        m.get("https://example.com/f", content=b"payload")
        with pytest.raises(FileNotFoundError):
            download_from_url_using_requests("https://example.com/f", target)
