import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from truss.contexts.image_builder import cache_warmer


@pytest.mark.parametrize("session_token", [None, "session-token"])
def test_parse_s3_credentials_file_valid(tmp_path, session_token):
    credentials = {
        "aws_access_key_id": "access-key",
        "aws_secret_access_key": "secret-key",
        "aws_region": "us-east-1",
    }
    if session_token is not None:
        credentials["aws_session_token"] = session_token
    credentials_file = tmp_path / "credentials.json"
    credentials_file.write_text(json.dumps(credentials))

    result = cache_warmer.parse_s3_credentials_file(str(credentials_file))

    assert result == cache_warmer.AWSCredentials(
        "access-key", "secret-key", "us-east-1", session_token
    )


@pytest.mark.parametrize(
    "missing_key", ["aws_access_key_id", "aws_secret_access_key", "aws_region"]
)
def test_parse_s3_credentials_file_invalid(tmp_path, missing_key):
    credentials = {
        "aws_access_key_id": "access-key",
        "aws_secret_access_key": "secret-key",
        "aws_region": "us-east-1",
    }
    del credentials[missing_key]
    credentials_file = tmp_path / "credentials.json"
    credentials_file.write_text(json.dumps(credentials))

    with pytest.raises(ValueError, match="Invalid AWS credentials file"):
        cache_warmer.parse_s3_credentials_file(str(credentials_file))


@pytest.mark.parametrize(
    ("path", "prefix", "expected"),
    [
        ("gs://bucket/file", "gs://", ("bucket", "file")),
        ("bucket/file", "gs://", ("bucket", "file")),
        ("bucket", "gs://", ("bucket", "")),
        ("gs://bucket/nested/file", "gs://", ("bucket", "nested/file")),
    ],
)
def test_split_path(path, prefix, expected):
    assert cache_warmer.split_path(path, prefix=prefix) == expected


@pytest.mark.parametrize("value", [None, "/usr/local/bin/b10cp"])
def test_b10cp_path(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(cache_warmer.B10CP_PATH_TRUSS_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(cache_warmer.B10CP_PATH_TRUSS_ENV_VAR_NAME, value)

    assert cache_warmer._b10cp_path() == value


def test_download_from_url_using_b10cp():
    download_to = Path("/tmp/model.bin")
    with patch.object(cache_warmer.subprocess, "Popen") as popen:
        result = cache_warmer._download_from_url_using_b10cp(
            "/usr/bin/b10cp", "https://example.com/model.bin", download_to
        )

    assert result is popen.return_value
    popen.assert_called_once_with(
        [
            "/usr/bin/b10cp",
            "-source",
            "https://example.com/model.bin",
            "-target",
            "/tmp/model.bin",
        ]
    )


@pytest.mark.parametrize(
    ("repo_name", "expected_type"),
    [
        ("gs://bucket", cache_warmer.GCSFile),
        ("s3://bucket", cache_warmer.S3File),
        ("org/model", cache_warmer.HuggingFaceFile),
    ],
)
def test_repository_file_from_file_dispatch(repo_name, expected_type):
    result = cache_warmer.RepositoryFile.from_file(repo_name, "weights.bin", "main")

    assert isinstance(result, expected_type)
    assert (result.repo_name, result.file_name, result.revision_name) == (
        repo_name,
        "weights.bin",
        "main",
    )


@pytest.mark.parametrize("secret", [None, "hf-secret"])
def test_huggingface_file_download_to_cache(monkeypatch, tmp_path, secret):
    secret_path = tmp_path / "hf_access_token"
    if secret is not None:
        secret_path.write_text(f" {secret} \n")
    monkeypatch.setattr(cache_warmer, "Path", lambda value: tmp_path / Path(value).name)
    file = cache_warmer.HuggingFaceFile("org/model", "weights.bin", "main")

    with patch.object(cache_warmer, "hf_hub_download") as download:
        file.download_to_cache()

    download.assert_called_once_with(
        "org/model", "weights.bin", revision="main", token=secret
    )


def test_huggingface_file_download_to_cache_missing_repository():
    file = cache_warmer.HuggingFaceFile("org/model", "weights.bin", "main")
    with patch.object(cache_warmer, "hf_hub_download", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="Hugging Face repository not found"):
            file.download_to_cache()


def test_download_file_using_b10cp_success(monkeypatch):
    process = MagicMock()
    download = MagicMock(return_value=process)
    monkeypatch.setattr(cache_warmer, "_download_from_url_using_b10cp", download)
    monkeypatch.setattr(cache_warmer, "_b10cp_path", lambda: "/bin/b10cp")

    cache_warmer.download_file_using_b10cp(
        "https://example.com/file", Path("/tmp/file"), "file"
    )

    download.assert_called_once_with(
        "/bin/b10cp", "https://example.com/file", Path("/tmp/file")
    )
    process.wait.assert_called_once_with()


def test_download_file_using_b10cp_missing_b10cp():
    with patch.object(
        cache_warmer,
        "_download_from_url_using_b10cp",
        side_effect=FileNotFoundError("b10cp"),
    ):
        with pytest.raises(
            RuntimeError, match="Failure due to file \\(file\\) not found"
        ):
            cache_warmer.download_file_using_b10cp(
                "https://example.com/file", Path("/tmp/file"), "file"
            )


def _patch_cache_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_warmer, "Path", lambda value: tmp_path / Path(value).name)


def test_gcs_file_download_private(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(cache_warmer, "repo_name", "gs://private-bucket", raising=False)
    monkeypatch.setattr(cache_warmer.os.path, "exists", lambda path: True)
    client = MagicMock()
    blob = MagicMock(name="weights.bin")
    blob.name = "weights.bin"
    blob.exists.return_value = True
    blob.generate_signed_url.return_value = "https://signed.example/file"
    client.bucket.return_value.blob.return_value = blob

    with (
        patch.object(
            cache_warmer.storage.Client,
            "from_service_account_json",
            return_value=client,
        ) as from_credentials,
        patch.object(cache_warmer, "download_file_using_b10cp") as download,
    ):
        cache_warmer.GCSFile(
            "gs://private-bucket", "weights.bin", "main"
        ).download_to_cache()

    from_credentials.assert_called_once_with(cache_warmer.GCS_CREDENTIALS)
    blob.generate_signed_url.assert_called_once()
    download.assert_called_once()
    assert download.call_args.args[0] == "https://signed.example/file"


def test_gcs_file_download_anonymous(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(cache_warmer, "repo_name", "gs://public-bucket", raising=False)
    monkeypatch.setattr(cache_warmer.os.path, "exists", lambda path: False)
    client = MagicMock()
    blob = MagicMock()
    blob.name = "nested/weights.bin"
    blob.exists.return_value = True
    client.bucket.return_value.blob.return_value = blob

    with (
        patch.object(
            cache_warmer.storage.Client, "create_anonymous_client", return_value=client
        ) as anonymous,
        patch.object(cache_warmer, "download_file_using_b10cp") as download,
    ):
        cache_warmer.GCSFile(
            "gs://public-bucket", "nested/weights.bin", "main"
        ).download_to_cache()

    anonymous.assert_called_once_with()
    blob.generate_signed_url.assert_not_called()
    download.assert_called_once_with(
        "https://storage.googleapis.com/public-bucket/nested/weights.bin",
        tmp_path / "public-bucket" / "nested/weights.bin",
        "nested/weights.bin",
    )


def test_gcs_file_download_missing_blob(monkeypatch, tmp_path):
    _patch_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(cache_warmer, "repo_name", "gs://bucket", raising=False)
    monkeypatch.setattr(cache_warmer.os.path, "exists", lambda path: False)
    client = MagicMock()
    blob = MagicMock()
    blob.name = "missing.bin"
    blob.exists.return_value = False
    client.bucket.return_value.blob.return_value = blob

    with patch.object(
        cache_warmer.storage.Client, "create_anonymous_client", return_value=client
    ):
        with pytest.raises(RuntimeError, match="File not found on GCS bucket"):
            cache_warmer.GCSFile(
                "gs://bucket", "missing.bin", "main"
            ).download_to_cache()


@pytest.mark.parametrize(
    ("credentials_exist", "object_name"),
    [(True, "weights.bin"), (False, "nested/weights.bin")],
)
def test_s3_file_download_success(
    monkeypatch, tmp_path, credentials_exist, object_name
):
    _patch_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(cache_warmer, "repo_name", "s3://bucket", raising=False)
    monkeypatch.setattr(cache_warmer, "file_name", object_name, raising=False)
    monkeypatch.setattr(cache_warmer.os.path, "exists", lambda path: credentials_exist)
    client = MagicMock()
    client.generate_presigned_url.return_value = "https://s3.example/file"
    credentials = cache_warmer.AWSCredentials("access", "secret", "region", "token")

    with (
        patch.object(cache_warmer, "boto3") as boto3,
        patch.object(
            cache_warmer, "parse_s3_credentials_file", return_value=credentials
        ) as parse_credentials,
        patch.object(cache_warmer, "download_file_using_b10cp") as download,
    ):
        boto3.client.return_value = client
        cache_warmer.S3File("s3://bucket", object_name, "main").download_to_cache()

    if credentials_exist:
        parse_credentials.assert_called_once_with(cache_warmer.S3_CREDENTIALS)
        kwargs = boto3.client.call_args.kwargs
        assert kwargs["aws_access_key_id"] == "access"
        assert kwargs["aws_session_token"] == "token"
    else:
        parse_credentials.assert_not_called()
        assert boto3.client.call_args.args == ("s3",)
    client.generate_presigned_url.assert_called_once_with(
        "get_object", Params={"Bucket": "bucket", "Key": object_name}, ExpiresIn=3600
    )
    download.assert_called_once_with(
        "https://s3.example/file", tmp_path / "bucket" / object_name, object_name
    )


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (NoCredentialsError(), "No AWS credentials found"),
        (
            ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
            ),
            "Client error when accessing the S3 bucket",
        ),
        (Exception("boom"), "File not found on S3 bucket: weights.bin"),
    ],
)
def test_s3_file_download_errors(monkeypatch, tmp_path, exception, message):
    _patch_cache_path(monkeypatch, tmp_path)
    monkeypatch.setattr(cache_warmer, "repo_name", "s3://bucket", raising=False)
    monkeypatch.setattr(cache_warmer, "file_name", "weights.bin", raising=False)
    monkeypatch.setattr(cache_warmer.os.path, "exists", lambda path: False)
    client = MagicMock()
    client.generate_presigned_url.side_effect = exception

    with patch.object(cache_warmer, "boto3") as boto3:
        boto3.client.return_value = client
        with pytest.raises(RuntimeError, match=message):
            cache_warmer.S3File(
                "s3://bucket", "weights.bin", "main"
            ).download_to_cache()


def test_download_file_dispatches_and_downloads():
    repository_file = MagicMock()
    with patch.object(
        cache_warmer.RepositoryFile, "from_file", return_value=repository_file
    ) as from_file:
        cache_warmer.download_file("org/model", "weights.bin", "main")

    from_file.assert_called_once_with("org/model", "weights.bin", "main")
    repository_file.download_to_cache.assert_called_once_with()
