import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from truss.util.error_utils import handle_client_error


def test_handle_client_error_no_credentials():
    with pytest.raises(RuntimeError, match="No AWS credentials found"):
        with handle_client_error("download"):
            raise NoCredentialsError()


def test_handle_client_error_client_error():
    error = ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "get")
    with pytest.raises(RuntimeError, match="AWS client error when upload"):
        with handle_client_error("upload"):
            raise error


def test_handle_client_error_unexpected():
    with pytest.raises(RuntimeError, match="Unexpected error"):
        with handle_client_error("sync"):
            raise ValueError("boom")
