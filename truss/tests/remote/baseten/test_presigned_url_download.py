from unittest import mock

import pytest
import requests

from truss.remote.baseten.api import PresignedUrlDownloadError

S3_INTERNAL_ERROR_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b"<Error><Code>InternalError</Code>"
    b"<Message>We encountered an internal error. Please try again.</Message>"
    b"<RequestId>8F3A1C2D9E4B7A61</RequestId></Error>"
)
TARBALL = b"\x1f\x8b" + b"x" * 100
URL = "https://bucket.s3.amazonaws.com/artifact.tgz?X-Amz-Signature=abc"


def _response(status_code: int, content: bytes, content_length=None):
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = content
    resp.headers["Content-Length"] = str(
        len(content) if content_length is None else content_length
    )
    return resp


@pytest.fixture(autouse=True)
def no_sleep():
    with mock.patch("tenacity.nap.time.sleep"):
        yield


def test_get_from_presigned_url_success(baseten_api):
    with mock.patch("requests.get", return_value=_response(200, TARBALL)) as get:
        assert baseten_api.get_from_presigned_url(URL) == TARBALL
    get.assert_called_once()
    assert get.call_args.args == (URL,)
    assert "timeout" in get.call_args.kwargs


def test_get_from_presigned_url_retries_s3_internal_error(baseten_api):
    responses = [
        _response(500, S3_INTERNAL_ERROR_XML),
        _response(503, b"slow down"),
        _response(200, TARBALL),
    ]
    with mock.patch("requests.get", side_effect=responses) as get:
        assert baseten_api.get_from_presigned_url(URL) == TARBALL
    assert get.call_count == 3


def test_get_from_presigned_url_retries_connection_reset(baseten_api):
    side_effects = [
        requests.exceptions.ConnectionError("Connection reset by peer"),
        requests.exceptions.ChunkedEncodingError("incomplete read"),
        _response(200, TARBALL),
    ]
    with mock.patch("requests.get", side_effect=side_effects) as get:
        assert baseten_api.get_from_presigned_url(URL) == TARBALL
    assert get.call_count == 3


def test_get_from_presigned_url_never_returns_error_body(baseten_api):
    responses = [_response(500, S3_INTERNAL_ERROR_XML)] * 5
    with mock.patch("requests.get", side_effect=responses) as get:
        with pytest.raises(PresignedUrlDownloadError, match="HTTP 500"):
            baseten_api.get_from_presigned_url(URL)
    assert get.call_count == 5


def test_get_from_presigned_url_reraises_after_retries_exhausted(baseten_api):
    err = requests.exceptions.ConnectionError("Connection reset by peer")
    with mock.patch("requests.get", side_effect=err) as get:
        with pytest.raises(requests.exceptions.ConnectionError):
            baseten_api.get_from_presigned_url(URL)
    assert get.call_count == 5


def test_get_from_presigned_url_does_not_retry_client_errors(baseten_api):
    with mock.patch("requests.get", return_value=_response(403, b"denied")) as get:
        with pytest.raises(PresignedUrlDownloadError, match="HTTP 403"):
            baseten_api.get_from_presigned_url(URL)
    get.assert_called_once()


def test_get_from_presigned_url_rejects_truncated_body(baseten_api):
    truncated = _response(200, TARBALL[:10], content_length=len(TARBALL))
    with mock.patch("requests.get", return_value=truncated):
        with pytest.raises(PresignedUrlDownloadError, match="truncated"):
            baseten_api.get_from_presigned_url(URL)
