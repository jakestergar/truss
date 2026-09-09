from unittest import mock

import pytest
import requests

from truss.remote.baseten.rest_client import RestAPIClient


@pytest.fixture
def client():
    return RestAPIClient(
        "https://api.example.com", lambda: {"Authorization": "Bearer token"}
    )


def test_rest_client_get(client):
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True}
    with mock.patch("requests.get", return_value=resp) as mock_get:
        result = client.get("models", {"foo": "bar"})
    mock_get.assert_called_once_with(
        "https://api.example.com/models",
        headers=mock.ANY,
        params={"foo": "bar"},
        timeout=(10, 60),
    )
    assert result == {"ok": True}


def test_rest_client_post(client):
    resp = mock.MagicMock()
    resp.status_code = 201
    resp.json.return_value = {"id": "123"}
    with mock.patch("requests.post", return_value=resp) as mock_post:
        result = client.post("models", {"name": "test"})
    mock_post.assert_called_once_with(
        "https://api.example.com/models",
        headers=mock.ANY,
        json={"name": "test"},
        timeout=(10, 60),
    )
    assert result == {"id": "123"}


def test_rest_client_delete(client):
    resp = mock.MagicMock()
    resp.status_code = 204
    resp.json.return_value = {}
    with mock.patch("requests.delete", return_value=resp) as mock_delete:
        result = client.delete("models/123")
    mock_delete.assert_called_once_with(
        "https://api.example.com/models/123", headers=mock.ANY, timeout=(10, 60)
    )
    assert result == {}


def test_rest_client_patch(client):
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"id": "123"}
    with mock.patch("requests.patch", return_value=resp) as mock_patch:
        result = client.patch("models/123", {"name": "new"})
    mock_patch.assert_called_once_with(
        "https://api.example.com/models/123",
        headers=mock.ANY,
        json={"name": "new"},
        timeout=(10, 60),
    )
    assert result == {"id": "123"}


def test_rest_client_handle_client_error_prints_message(client):
    resp = mock.MagicMock()
    resp.status_code = 400
    resp.json.return_value = {"message": "bad request"}
    resp.raise_for_status.side_effect = requests.HTTPError("400")
    with mock.patch("builtins.print") as mock_print:
        with pytest.raises(requests.HTTPError):
            client._handle_error(resp)
    mock_print.assert_called_once_with("Client error: bad request")


def test_rest_client_handle_client_error_suppressed(client):
    client.suppress_error_print = True
    resp = mock.MagicMock()
    resp.status_code = 400
    resp.json.return_value = {"message": "bad request"}
    resp.raise_for_status.side_effect = requests.HTTPError("400")
    with mock.patch("builtins.print") as mock_print:
        with pytest.raises(requests.HTTPError):
            client._handle_error(resp)
    mock_print.assert_not_called()


def test_rest_client_handle_non_json_4xx(client):
    resp = mock.MagicMock()
    resp.status_code = 404
    resp.json.side_effect = ValueError("not json")
    resp.raise_for_status.side_effect = requests.HTTPError("404")
    with pytest.raises(requests.HTTPError):
        client._handle_error(resp)
