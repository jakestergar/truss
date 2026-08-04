from unittest import mock

import pytest
import requests
from requests import Response

from truss.remote.baseten import user_agent
from truss.remote.baseten.rest_client import _DEFAULT_TIMEOUT_SEC, RestAPIClient


def mock_response(status_code=200, json_data=None, json_raises=False):
    response = Response()
    response.status_code = status_code
    if json_raises:
        response.json = mock.Mock(side_effect=ValueError("no json"))
    else:
        response.json = mock.Mock(return_value=json_data or {"data": "ok"})
    return response


@pytest.fixture
def rest_client():
    return RestAPIClient(
        "https://app.test.com", lambda: {"Authorization": "Api-Key token"}
    )


def test_headers_include_user_agent_and_provider_headers(rest_client):
    headers = rest_client._headers()
    assert headers["Authorization"] == "Api-Key token"
    assert headers["User-Agent"] == user_agent.user_agent_header()


def test_headers_calls_provider_on_every_request():
    header_provider = mock.Mock(side_effect=[{"Authorization": "a"}, {"X-Token": "b"}])
    client = RestAPIClient("https://app.test.com", header_provider)
    assert client._headers()["Authorization"] == "a"
    assert client._headers()["X-Token"] == "b"
    assert header_provider.call_count == 2


def test_suppress_error_print_defaults_to_false(rest_client):
    assert rest_client.suppress_error_print is False
    assert rest_client.base_url == "https://app.test.com"


@pytest.mark.parametrize(
    "method, kwargs, expected_payload_key",
    [
        ("get", {"url_params": {"page": "1"}}, "params"),
        ("post", {"body": {"a": 1}}, "json"),
        ("delete", {}, None),
        ("patch", {"body": {"a": 1}}, "json"),
    ],
)
def test_request_methods_return_json_and_pass_headers(
    rest_client, method, kwargs, expected_payload_key
):
    response = mock_response(json_data={"data": "ok"})
    with mock.patch(f"requests.{method}", return_value=response) as mock_request:
        result = getattr(rest_client, method)("v1/models", **kwargs)

    assert result == {"data": "ok"}
    call_kwargs = mock_request.call_args[1]
    assert mock_request.call_args[0][0] == "https://app.test.com/v1/models"
    assert call_kwargs["headers"]["Authorization"] == "Api-Key token"
    assert call_kwargs["headers"]["User-Agent"] == user_agent.user_agent_header()
    assert call_kwargs["timeout"] == _DEFAULT_TIMEOUT_SEC
    if expected_payload_key is not None:
        assert call_kwargs[expected_payload_key] == list(kwargs.values())[0]


def test_get_defaults_to_empty_url_params(rest_client):
    with mock.patch("requests.get", return_value=mock_response()) as mock_get:
        rest_client.get("v1/models")

    assert mock_get.call_args[1]["params"] == {}


@pytest.mark.parametrize("body", [None, {}, [], {"nested": {"a": [1, 2]}}])
def test_post_forwards_body_verbatim(rest_client, body):
    with mock.patch("requests.post", return_value=mock_response()) as mock_post:
        rest_client.post("v1/models", body)

    assert mock_post.call_args[1]["json"] == body


@pytest.mark.parametrize("method", ["get", "post", "delete", "patch"])
def test_client_error_prints_message_and_raises(rest_client, method, capsys):
    response = mock_response(status_code=400, json_data={"message": "bad request"})
    args = [] if method in ("get", "delete") else [{"a": 1}]
    with mock.patch(f"requests.{method}", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError):
            getattr(rest_client, method)("v1/models", *args)

    assert "Client error: bad request" in capsys.readouterr().out


def test_client_error_message_suppressed(rest_client, capsys):
    rest_client.suppress_error_print = True
    response = mock_response(status_code=400, json_data={"message": "bad request"})
    with mock.patch("requests.get", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError):
            rest_client.get("v1/models")

    assert capsys.readouterr().out == ""


def test_client_error_without_message_key_raises_silently(rest_client, capsys):
    response = mock_response(status_code=404, json_data={"detail": "missing"})
    with mock.patch("requests.get", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError):
            rest_client.get("v1/models")

    assert capsys.readouterr().out == ""


def test_client_error_with_non_json_body_raises_silently(rest_client, capsys):
    response = mock_response(status_code=422, json_raises=True)
    with mock.patch("requests.post", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError):
            rest_client.post("v1/models", {"a": 1})

    assert capsys.readouterr().out == ""


def test_server_error_does_not_inspect_body(rest_client, capsys):
    response = mock_response(status_code=500, json_data={"message": "boom"})
    with mock.patch("requests.delete", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError):
            rest_client.delete("v1/models/1")

    assert capsys.readouterr().out == ""
    response.json.assert_not_called()


@pytest.mark.parametrize("status_code", [200, 201, 204, 302])
def test_non_error_statuses_are_not_raised(rest_client, status_code):
    response = mock_response(status_code=status_code)
    with mock.patch("requests.get", return_value=response):
        assert rest_client.get("v1/models") == {"data": "ok"}
