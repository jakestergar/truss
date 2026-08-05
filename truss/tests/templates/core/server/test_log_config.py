import json
import logging

import pytest

from truss.templates.shared import log_config


@pytest.fixture
def request_ids():
    request_token = log_config.request_id_context.set("request-1")
    chain_token = log_config.chain_request_id_context.set("chain-1")
    try:
        yield
    finally:
        log_config.request_id_context.reset(request_token)
        log_config.chain_request_id_context.reset(chain_token)


def _record(msg, args=(), name="uvicorn.access"):
    return logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


@pytest.mark.parametrize(
    "msg, expected",
    [
        ('127.0.0.1 - "GET / HTTP/1.1" 200', False),
        ('127.0.0.1 - "GET /v1/models/model HTTP/1.1" 200', False),
        ('127.0.0.1 - "GET /v1/models/model/loaded HTTP/1.1" 200', False),
        ('127.0.0.1 - "POST /v1/models/model:predict HTTP/1.1" 200', True),
    ],
)
def test_health_check_filter(msg, expected):
    assert log_config._HealthCheckFilter().filter(_record(msg)) is expected


@pytest.mark.parametrize(
    "msg, expected", [("connection open", False), ("connection closed", True)]
)
def test_websocket_open_filter(msg, expected):
    assert log_config._WebsocketOpenFilter().filter(_record(msg)) is expected


@pytest.mark.parametrize(
    "msg, expected",
    [('127.0.0.1 - "GET /metrics HTTP/1.1" 200', False), ("something else", True)],
)
def test_metrics_filter(msg, expected):
    assert log_config._MetricsFilter().filter(_record(msg)) is expected


def test_access_json_formatter_rewrites_uvicorn_access_records(request_ids):
    formatter = log_config._AccessJsonFormatter("%(asctime)s %(levelname)s %(message)s")
    record = _record(
        '%s - "%s %s HTTP/%s" %d', args=("127.0.0.1", "GET", "/a%20b", "1.1", 200)
    )

    parsed = json.loads(formatter.format(record))

    assert parsed["message"] == "Handled request: GET /a b HTTP/1.1 200"
    assert parsed["request_id"] == "request-1"
    assert parsed["chain_request_id"] == "chain-1"


def test_access_json_formatter_leaves_other_records_untouched():
    formatter = log_config._AccessJsonFormatter("%(asctime)s %(levelname)s %(message)s")
    record = _record("plain message", name="uvicorn.error")

    parsed = json.loads(formatter.format(record))

    assert parsed["message"] == "plain message"
    assert "request_id" not in parsed
    assert "chain_request_id" not in parsed


def test_default_json_formatter_adds_request_ids(request_ids):
    formatter = log_config._DefaultJsonFormatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    parsed = json.loads(formatter.format(_record("hello", name="root")))

    assert parsed["message"] == "hello"
    assert parsed["request_id"] == "request-1"
    assert parsed["chain_request_id"] == "chain-1"


def test_default_json_formatter_without_request_ids():
    formatter = log_config._DefaultJsonFormatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    parsed = json.loads(formatter.format(_record("hello", name="root")))

    assert "request_id" not in parsed
    assert "chain_request_id" not in parsed


def test_access_formatter_rewrites_uvicorn_access_records():
    formatter = log_config._AccessFormatter("%(message)s")
    record = _record(
        '%s - "%s %s HTTP/%s" %d', args=("127.0.0.1", "GET", "/a%20b", "1.1", 200)
    )

    assert formatter.format(record) == "Handled request - GET /a b HTTP/1.1 200"


def test_access_formatter_leaves_other_records_untouched():
    formatter = log_config._AccessFormatter("%(message)s")

    assert formatter.format(_record("plain", name="root")) == "plain"


def test_make_log_config_uses_json_formatters_by_default(monkeypatch):
    monkeypatch.delenv("DISABLE_JSON_LOGGING", raising=False)

    config = log_config.make_log_config("DEBUG")

    assert config["formatters"]["default_formatter"]["()"] is (
        log_config._DefaultJsonFormatter
    )
    assert config["formatters"]["access_formatter"]["()"] is (
        log_config._AccessJsonFormatter
    )
    assert config["loggers"]["uvicorn"]["level"] == "DEBUG"
    assert config["root"]["level"] == "DEBUG"


def test_make_log_config_uses_plain_formatters_when_json_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_JSON_LOGGING", "true")

    config = log_config.make_log_config("INFO")

    assert "()" not in config["formatters"]["default_formatter"]
    assert config["formatters"]["default_formatter"]["datefmt"] == (
        log_config.LOCAL_DATE_FORMAT
    )
    assert config["formatters"]["access_formatter"]["()"] is log_config._AccessFormatter


def test_make_log_config_filters_health_checks_and_metrics_on_access_logger():
    config = log_config.make_log_config("INFO")

    assert config["loggers"]["uvicorn.access"]["filters"] == [
        "health_check_filter",
        "metrics_filter",
    ]
    assert config["loggers"]["uvicorn.error"]["filters"] == ["websocket_filter"]
    assert config["filters"]["health_check_filter"]["()"] is (
        log_config._HealthCheckFilter
    )
