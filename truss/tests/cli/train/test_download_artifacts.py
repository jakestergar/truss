import io
import json
import os
import tarfile
from pathlib import Path
from unittest.mock import Mock

import click
import pytest

from truss.cli.train.core import (
    _generate_job_artifact_name,
    download_checkpoint_artifacts,
    download_training_job_data,
)

JOB_ID = "job123"
PROJECT_ID = "proj456"


def _make_job(project_name: str = "my project", job_id: str = JOB_ID) -> dict:
    return {
        "id": job_id,
        "current_status": "TRAINING_JOB_COMPLETED",
        "training_project": {"id": PROJECT_ID, "name": project_name},
    }


def _make_remote(content: bytes = b"artifact-bytes", job: dict = None) -> Mock:
    job = job if job is not None else _make_job()
    api = Mock()
    api.search_training_jobs.return_value = [job]
    api.get_training_job_presigned_url.return_value = "https://example.com/presigned"
    api.get_from_presigned_url.return_value = content
    remote = Mock()
    remote.api = api
    return remote


def _tarball(members: list) -> bytes:
    """Build an in-memory .tar.gz. `members` is a list of (TarInfo, bytes|None)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info, data in members:
            if data is None:
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _file_member(name: str, data: bytes = b"hello") -> tuple:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    return info, data


# ---------------------------------------------------------------------------
# _generate_job_artifact_name
# ---------------------------------------------------------------------------


def test_generate_job_artifact_name():
    assert _generate_job_artifact_name("proj", "job1") == "proj_job1"
    # spaces are NOT handled here, only by the callers
    assert _generate_job_artifact_name("my proj", "job1") == "my proj_job1"


# ---------------------------------------------------------------------------
# download_training_job_data - unzip=False
# ---------------------------------------------------------------------------


def test_download_no_unzip_writes_bytes(tmp_path):
    remote = _make_remote(content=b"tarball-content", job=_make_job("proj"))

    result = download_training_job_data(
        remote_provider=remote,
        job_id=JOB_ID,
        target_directory=str(tmp_path),
        unzip=False,
    )

    assert result == tmp_path / f"proj_{JOB_ID}.tgz"
    assert result.read_bytes() == b"tarball-content"
    remote.api.get_training_job_presigned_url.assert_called_once_with(
        project_id=PROJECT_ID, job_id=JOB_ID
    )
    remote.api.get_from_presigned_url.assert_called_once_with(
        "https://example.com/presigned"
    )


def test_download_no_unzip_spaces_in_project_name_become_dashes(tmp_path):
    remote = _make_remote(job=_make_job("my cool project"))

    result = download_training_job_data(remote, JOB_ID, str(tmp_path), unzip=False)

    assert result == tmp_path / f"my-cool-project_{JOB_ID}.tgz"
    assert result.exists()


def test_download_no_unzip_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    remote = _make_remote(job=_make_job("proj"))

    result = download_training_job_data(remote, JOB_ID, None, unzip=False)

    assert result == Path(tmp_path) / f"proj_{JOB_ID}.tgz"
    assert result.exists()


def test_download_no_unzip_creates_nested_target_directory(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    remote = _make_remote(job=_make_job("proj"))

    result = download_training_job_data(remote, JOB_ID, str(target), unzip=False)

    assert result.parent == target
    assert result.exists()


def test_download_no_unzip_relative_target_directory_is_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    remote = _make_remote(job=_make_job("proj"))

    result = download_training_job_data(remote, JOB_ID, "sub/dir", unzip=False)

    assert result.is_absolute()
    assert result == tmp_path.resolve() / "sub" / "dir" / f"proj_{JOB_ID}.tgz"
    assert result.exists()


def test_download_no_unzip_target_directory_containing_space_is_mangled(tmp_path):
    """BUG: the space-stripping is applied to the whole path, not just the file name.

    `download_training_job_data` does
    `target_path = Path(str(target_path).replace(" ", "-"))`, which also rewrites any
    space in the *user supplied* target directory. A user who downloads into e.g.
    `~/My Trainings` gets a FileNotFoundError (the mangled `My-Trainings` directory was
    never created) instead of their artifact.

    This test asserts the current (wrong) behavior.
    """
    target = tmp_path / "My Trainings"
    remote = _make_remote(job=_make_job("proj"))

    with pytest.raises(FileNotFoundError):
        download_training_job_data(remote, JOB_ID, str(target), unzip=False)

    # the directory that was created is empty; nothing was written anywhere
    assert list(target.iterdir()) == []
    assert not (tmp_path / "My-Trainings").exists()


# ---------------------------------------------------------------------------
# download_training_job_data - unzip=True
# ---------------------------------------------------------------------------


def test_download_unzip_happy_path(tmp_path):
    content = _tarball([_file_member("model/weights.bin", b"weights")])
    remote = _make_remote(content=content, job=_make_job("proj"))

    result = download_training_job_data(remote, JOB_ID, str(tmp_path), unzip=True)

    assert result == tmp_path / f"proj_{JOB_ID}"
    assert (result / "model" / "weights.bin").read_bytes() == b"weights"


def test_download_unzip_spaces_in_project_name_become_dashes(tmp_path):
    content = _tarball([_file_member("a.txt")])
    remote = _make_remote(content=content, job=_make_job("my cool project"))

    result = download_training_job_data(remote, JOB_ID, str(tmp_path), unzip=True)

    assert result == tmp_path / f"my-cool-project_{JOB_ID}"
    assert (result / "a.txt").exists()


def test_download_unzip_existing_directory_raises(tmp_path):
    content = _tarball([_file_member("a.txt")])
    remote = _make_remote(content=content, job=_make_job("proj"))
    (tmp_path / f"proj_{JOB_ID}").mkdir()

    with pytest.raises(click.ClickException) as exc_info:
        download_training_job_data(remote, JOB_ID, str(tmp_path), unzip=True)

    assert "already exists" in str(exc_info.value)


def test_download_unzip_invalid_tarball_raises_raw_read_error(tmp_path):
    """BUG (usability): a non-tarball payload surfaces a bare `tarfile.ReadError`.

    The download is not wrapped in try/except, so a truncated/corrupt/HTML-error-page
    download crashes the CLI with `tarfile.ReadError: file could not be opened
    successfully` (and, in the CLI, a full traceback) rather than an actionable
    ClickException telling the user the artifact download was corrupt. It also leaves
    behind an empty output directory.

    This test asserts the current behavior.
    """
    remote = _make_remote(
        content=b"this is definitely not a tarball", job=_make_job("proj")
    )

    with pytest.raises(tarfile.ReadError):
        download_training_job_data(remote, JOB_ID, str(tmp_path), unzip=True)

    # empty directory left behind, so an immediate retry fails with "already exists"
    leftover = tmp_path / f"proj_{JOB_ID}"
    assert leftover.is_dir()
    assert list(leftover.iterdir()) == []
    with pytest.raises(click.ClickException):
        download_training_job_data(remote, JOB_ID, str(tmp_path), unzip=True)


# ---------------------------------------------------------------------------
# SECURITY: tar.extractall() without filter= -> path traversal (CVE-2007-4559)
# ---------------------------------------------------------------------------


def test_download_unzip_relative_path_traversal_escapes_target_dir(tmp_path):
    """SECURITY BUG: `tar.extractall(path=unzip_dir)` is called with no `filter=`.

    A member named `../pwned.txt` is written *outside* the unzip directory. On the
    Pythons this repo supports (>=3.9,<3.15) the default extraction filter is
    "fully_trusted" (Python <3.14; 3.12/3.13 only emit a DeprecationWarning), so the
    traversal succeeds. This test asserts the current (insecure) behavior.
    """
    content = _tarball([_file_member("../pwned.txt", b"owned")])
    remote = _make_remote(content=content, job=_make_job("proj"))
    target = tmp_path / "downloads"

    result = download_training_job_data(remote, JOB_ID, str(target), unzip=True)

    escaped = target / "pwned.txt"
    assert result == target / f"proj_{JOB_ID}"
    assert escaped.exists(), "expected traversal to be blocked, but it is not"
    assert escaped.read_bytes() == b"owned"
    assert not (result / "pwned.txt").exists()


def test_download_unzip_traversal_overwrites_existing_file(tmp_path):
    """SECURITY BUG: traversal can clobber pre-existing user files."""
    content = _tarball([_file_member("../../victim.txt", b"malicious")])
    remote = _make_remote(content=content, job=_make_job("proj"))
    victim = tmp_path / "victim.txt"
    victim.write_text("important user data")
    target = tmp_path / "downloads"

    download_training_job_data(remote, JOB_ID, str(target), unzip=True)

    assert victim.read_bytes() == b"malicious"


def test_download_unzip_absolute_member_path(tmp_path):
    """Absolute member paths: record what actually lands on disk."""
    absolute_target = tmp_path / "abs_target.txt"
    content = _tarball([_file_member(str(absolute_target), b"absolute")])
    remote = _make_remote(content=content, job=_make_job("proj"))
    target = tmp_path / "downloads"

    result = download_training_job_data(remote, JOB_ID, str(target), unzip=True)

    # os.path.join(unzip_dir, "/abs/path") == "/abs/path": the member escapes entirely.
    assert absolute_target.read_bytes() == b"absolute"
    assert not any(result.rglob("abs_target.txt"))


def test_download_unzip_symlink_member_escapes(tmp_path):
    """SECURITY BUG: symlink members are extracted verbatim.

    A symlink pointing outside the unzip directory is created as-is, so a subsequent
    member written "through" it lands outside the target directory.
    """
    outside = tmp_path / "outside"
    outside.mkdir()

    link = tarfile.TarInfo(name="escape")
    link.type = tarfile.SYMTYPE
    link.linkname = str(outside)

    content = _tarball([(link, None), _file_member("escape/planted.txt", b"planted")])
    remote = _make_remote(content=content, job=_make_job("proj"))
    target = tmp_path / "downloads"

    result = download_training_job_data(remote, JOB_ID, str(target), unzip=True)

    assert (result / "escape").is_symlink()
    assert os.readlink(result / "escape") == str(outside)
    assert (outside / "planted.txt").read_bytes() == b"planted"


# ---------------------------------------------------------------------------
# Server-controlled project_name flows straight into the output path
# ---------------------------------------------------------------------------


def test_project_name_with_slash_escapes_target_directory_no_unzip(tmp_path):
    """BUG: `project_name` comes from the API and is only space-sanitized.

    A project named `../evil` makes the .tgz land outside the requested directory.
    """
    remote = _make_remote(job=_make_job("../evil"))
    target = tmp_path / "downloads"

    result = download_training_job_data(remote, JOB_ID, str(target), unzip=False)

    assert result == target / f"../evil_{JOB_ID}.tgz"
    assert (tmp_path / f"evil_{JOB_ID}.tgz").exists()
    assert not (target / f"evil_{JOB_ID}.tgz").exists()


def test_project_name_with_nested_slash_fails_no_unzip(tmp_path):
    """A project name containing `/` points at a nonexistent subdirectory."""
    remote = _make_remote(job=_make_job("team/proj"))
    target = tmp_path / "downloads"

    with pytest.raises(FileNotFoundError):
        download_training_job_data(remote, JOB_ID, str(target), unzip=False)

    assert not (target / "team").exists()


def test_project_name_with_leading_dash_no_unzip(tmp_path):
    """A leading `-` is preserved, producing a file that looks like a CLI flag."""
    remote = _make_remote(job=_make_job("-rf"))

    result = download_training_job_data(remote, JOB_ID, str(tmp_path), unzip=False)

    assert result.name == f"-rf_{JOB_ID}.tgz"
    assert result.exists()


def test_project_name_with_slash_unzip_creates_directory_outside_target(tmp_path):
    content = _tarball([_file_member("a.txt", b"data")])
    remote = _make_remote(content=content, job=_make_job("../evil"))
    target = tmp_path / "downloads"

    result = download_training_job_data(remote, JOB_ID, str(target), unzip=True)

    assert result == target / f"../evil_{JOB_ID}"
    assert (tmp_path / f"evil_{JOB_ID}" / "a.txt").read_bytes() == b"data"


# ---------------------------------------------------------------------------
# download_checkpoint_artifacts
# ---------------------------------------------------------------------------


def _make_checkpoint_remote(
    checkpoints: list = None, job: dict = None, latest_job: dict = None
) -> Mock:
    api = Mock()
    api.get_training_job_checkpoint_presigned_url.return_value = (
        checkpoints if checkpoints is not None else [{"checkpoint_id": "ckpt-1"}]
    )

    def _search(**kwargs):
        if "job_id" in kwargs:
            return [job if job is not None else _make_job()]
        return [latest_job if latest_job is not None else _make_job()]

    api.search_training_jobs.side_effect = _search
    remote = Mock()
    remote.api = api
    return remote


def test_download_checkpoint_artifacts_with_job_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job = _make_job("proj", job_id="job-explicit")
    remote = _make_checkpoint_remote(
        checkpoints=[{"checkpoint_id": "ckpt-1", "url": "https://x"}], job=job
    )

    result = download_checkpoint_artifacts(remote, "job-explicit")

    assert result == Path(tmp_path) / "proj_job-explicit_checkpoints.json"
    payload = json.loads(result.read_text())
    assert set(payload) == {"timestamp", "job", "checkpoint_artifacts"}
    assert payload["job"] == job
    assert payload["checkpoint_artifacts"] == [
        {"checkpoint_id": "ckpt-1", "url": "https://x"}
    ]
    remote.api.search_training_jobs.assert_called_once_with(job_id="job-explicit")
    remote.api.get_training_job_checkpoint_presigned_url.assert_called_once_with(
        project_id=PROJECT_ID, job_id="job-explicit", page_size=1000
    )


def test_download_checkpoint_artifacts_falls_back_to_latest_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    latest = _make_job("proj", job_id="job-latest")
    remote = _make_checkpoint_remote(latest_job=latest)

    result = download_checkpoint_artifacts(remote, None)

    assert result.name == "proj_job-latest_checkpoints.json"
    remote.api.search_training_jobs.assert_called_once_with(
        order_by=[{"field": "created_at", "order": "desc"}]
    )


def test_download_checkpoint_artifacts_no_jobs_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    remote = Mock()
    remote.api = Mock()
    remote.api.search_training_jobs.return_value = []

    with pytest.raises(click.ClickException) as exc_info:
        download_checkpoint_artifacts(remote, None)

    assert "No training jobs found" in str(exc_info.value)


def test_download_checkpoint_artifacts_unknown_job_id_raises_runtime_error(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    remote = Mock()
    remote.api = Mock()
    remote.api.search_training_jobs.return_value = []

    with pytest.raises(RuntimeError, match="No training job found with ID: nope"):
        download_checkpoint_artifacts(remote, "nope")


def test_download_checkpoint_artifacts_empty_checkpoints_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    remote = _make_checkpoint_remote(checkpoints=[])

    with pytest.raises(click.ClickException) as exc_info:
        download_checkpoint_artifacts(remote, JOB_ID)

    assert "No checkpoints found" in str(exc_info.value)
    assert list(Path(tmp_path).iterdir()) == []


def test_download_checkpoint_artifacts_sanitizes_spaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    remote = _make_checkpoint_remote(job=_make_job("my cool project"))

    result = download_checkpoint_artifacts(remote, JOB_ID)

    assert result.name == f"my-cool-project_{JOB_ID}_checkpoints.json"


def test_download_checkpoint_artifacts_project_name_with_slash_escapes_cwd(
    tmp_path, monkeypatch
):
    """BUG: only spaces are sanitized, so a `../` project name escapes the cwd."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    remote = _make_checkpoint_remote(job=_make_job("../evil"))

    result = download_checkpoint_artifacts(remote, JOB_ID)

    assert result == workdir / f"../evil_{JOB_ID}_checkpoints.json"
    assert (tmp_path / f"evil_{JOB_ID}_checkpoints.json").exists()
