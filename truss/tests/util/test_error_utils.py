import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from truss.util.error_utils import handle_client_error


def test_handle_client_error_passes_through_on_success():
    with handle_client_error("listing buckets"):
        result = "ok"

    assert result == "ok"


def test_handle_client_error_wraps_missing_credentials():
    with pytest.raises(RuntimeError, match="No AWS credentials found for uploading"):
        with handle_client_error("uploading"):
            raise NoCredentialsError()


def test_handle_client_error_wraps_client_error():
    error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
    )

    with pytest.raises(RuntimeError, match="AWS client error when downloading"):
        with handle_client_error("downloading"):
            raise error


def test_handle_client_error_wraps_unexpected_error():
    with pytest.raises(RuntimeError, match="Unexpected error `boom` during `syncing`"):
        with handle_client_error("syncing"):
            raise ValueError("boom")
