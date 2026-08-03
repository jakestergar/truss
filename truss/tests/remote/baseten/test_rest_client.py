from unittest.mock import Mock, patch

import pytest

from truss.remote.baseten.rest_client import RestAPIClient


@pytest.fixture
def client():
    return RestAPIClient("https://api.example", lambda: {"Authorization": "token"})


def response(status=200, payload=None):
    result = Mock(status_code=status)
    result.json.return_value = payload if payload is not None else {"ok": True}
    return result


@pytest.mark.parametrize(
    ("method", "path", "payload", "request_method"),
    [
        ("get", "models", None, "get"),
        ("post", "models", {"name": "x"}, "post"),
        ("delete", "models/1", None, "delete"),
        ("patch", "models/1", {"name": "y"}, "patch"),
    ],
)
def test_http_methods_add_headers_and_timeout(
    client, method, path, payload, request_method
):
    resp = response(payload={"result": method})
    with patch(
        f"truss.remote.baseten.rest_client.requests.{request_method}", return_value=resp
    ) as request:
        result = (
            getattr(client, method)(path)
            if payload is None
            else getattr(client, method)(path, payload)
        )
    assert result == {"result": method}
    kwargs = request.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "token"
    assert "User-Agent" in kwargs["headers"]
    assert kwargs["timeout"] == (10, 60)
    if method == "get":
        assert kwargs["params"] == {}
    if method in {"post", "patch"}:
        assert kwargs["json"] == payload


def test_headers_does_not_mutate_provider_result(client):
    with patch(
        "truss.remote.baseten.rest_client.with_user_agent",
        side_effect=lambda value: {**value, "User-Agent": "test"},
    ) as with_user_agent:
        result = client._headers()
    with_user_agent.assert_called_once_with({"Authorization": "token"})
    assert result["Authorization"] == "token"


def test_client_error_prints_message_and_raises(client, capsys):
    resp = response(status=400, payload={"message": "bad request"})
    resp.raise_for_status.side_effect = RuntimeError("bad")
    with pytest.raises(RuntimeError, match="bad"):
        client._handle_error(resp)
    assert "Client error: bad request" in capsys.readouterr().out


def test_client_error_can_suppress_message(client, capsys):
    client.suppress_error_print = True
    resp = response(status=404, payload={"message": "missing"})
    resp.raise_for_status.side_effect = RuntimeError("missing")
    with pytest.raises(RuntimeError):
        client._handle_error(resp)
    assert capsys.readouterr().out == ""


def test_client_error_with_invalid_json_and_non_client_error(client):
    resp = response(status=500)
    resp.json.side_effect = ValueError
    resp.raise_for_status.side_effect = RuntimeError("server")
    with pytest.raises(RuntimeError):
        client._handle_error(resp)
    resp = response(status=200)
    client._handle_error(resp)
    resp.raise_for_status.assert_called_once()
