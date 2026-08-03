from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import rich_click as click

from truss.cli.remote_cli import (
    NonEmptyValidator,
    format_available_teams,
    get_team_id_from_name,
    inquire_model_name,
    inquire_remote_config,
    inquire_remote_name,
    inquire_team,
)
from truss.remote.baseten.custom_types import TeamType
from truss.remote.remote_factory import AuthType


def test_non_empty_validator_rejects_whitespace():
    with pytest.raises(Exception, match="non-empty"):
        NonEmptyValidator().validate(SimpleNamespace(text=" \t"))


def test_inquire_remote_config_api_key():
    with (
        patch("truss.cli.remote_cli.inquirer.select") as select,
        patch("truss.cli.remote_cli.inquirer.secret") as secret,
    ):
        select.return_value.execute.return_value = "api_key"
        secret.return_value.execute.return_value = "secret"
        config = inquire_remote_config(
            remote_name="custom", remote_url="https://remote"
        )
    assert config.name == "custom"
    assert config.configs["auth_type"] is AuthType.API_KEY
    assert config.configs["api_key"] == "secret"


def test_inquire_remote_config_oauth_and_error():
    credential = SimpleNamespace(
        access_token="access", refresh_token="refresh", expires_at=42
    )
    with (
        patch("truss.cli.remote_cli.inquirer.select") as select,
        patch(
            "truss.cli.remote_cli.oauth.run_device_flow", return_value=credential
        ) as flow,
        patch("truss.cli.remote_cli.resolve_rest_api_url", return_value="api"),
    ):
        select.return_value.execute.return_value = "browser"
        config = inquire_remote_config(remote_url="https://remote")
    flow.assert_called_once_with("api")
    assert config.configs["auth_type"] is AuthType.OAUTH
    assert config.configs["oauth_expires_at"] == "42"

    with (
        patch("truss.cli.remote_cli.inquirer.select") as select,
        patch("truss.cli.remote_cli.inquirer.secret") as secret,
    ):
        select.return_value.execute.return_value = "api_key"
        secret.return_value.execute.return_value = "api-key"
        assert inquire_remote_config().configs["auth_type"] is AuthType.API_KEY


@pytest.mark.parametrize(
    ("remotes", "interactive", "allow_create", "expected"),
    [
        (["one"], False, True, "one"),
        ([], False, False, click.ClickException),
        ([], False, True, click.UsageError),
        (["one", "two"], False, True, click.UsageError),
    ],
)
def test_inquire_remote_name_noninteractive(
    remotes, interactive, allow_create, expected
):
    with (
        patch(
            "truss.cli.remote_cli.RemoteFactory.get_available_config_names",
            return_value=remotes,
        ),
        patch("truss.cli.remote_cli.check_is_interactive", return_value=interactive),
    ):
        if isinstance(expected, str):
            assert inquire_remote_name(allow_create=allow_create) == expected
        else:
            with pytest.raises(expected):
                inquire_remote_name(allow_create=allow_create)


def test_inquire_remote_name_creates_config_and_selects_multiple():
    config = Mock(name="new")
    with (
        patch(
            "truss.cli.remote_cli.RemoteFactory.get_available_config_names",
            return_value=[],
        ),
        patch("truss.cli.remote_cli.check_is_interactive", return_value=True),
        patch("truss.cli.remote_cli.inquire_remote_config", return_value=config),
        patch("truss.cli.remote_cli.RemoteFactory.update_remote_config") as update,
    ):
        assert inquire_remote_name() == config.name
    update.assert_called_once_with(config)

    with (
        patch(
            "truss.cli.remote_cli.RemoteFactory.get_available_config_names",
            return_value=["z", "a"],
        ),
        patch("truss.cli.remote_cli.check_is_interactive", return_value=True),
        patch("truss.cli.remote_cli.inquirer.select") as select,
    ):
        select.return_value.execute.return_value = "a"
        assert inquire_remote_name() == "a"


def test_model_and_team_helpers():
    with patch("truss.cli.remote_cli.inquirer.text") as text:
        text.return_value.execute.return_value = "model name"
        assert inquire_model_name() == "model name"
    teams = {
        "Alpha": TeamType(id="1", name="Alpha", default=False),
        "Default": TeamType(id="2", name="Default", default=True),
    }
    assert get_team_id_from_name(teams, "Alpha") == "1"
    assert get_team_id_from_name(teams, "missing") is None
    assert format_available_teams(teams) == "Alpha, Default"
    assert format_available_teams({}) == "none"


def test_inquire_team_none_and_noninteractive_error():
    assert inquire_team() is None
    teams = {"z": TeamType(id="1", name="z", default=False)}
    with patch("truss.cli.remote_cli.check_is_interactive", return_value=False):
        with pytest.raises(click.UsageError, match="available: z"):
            inquire_team(teams)


def test_inquire_team_sorts_default_first_and_returns_value():
    teams = {
        "zeta": TeamType(id="1", name="zeta", default=False),
        "Alpha": TeamType(id="2", name="Alpha", default=False),
        "default": TeamType(id="3", name="default", default=True),
    }
    with (
        patch("truss.cli.remote_cli.check_is_interactive", return_value=True),
        patch("truss.cli.remote_cli.inquirer.select") as select,
    ):
        select.return_value.execute.return_value = "default"
        assert inquire_team(teams) == "default"
    choices = select.call_args.kwargs["choices"]
    assert [choice.value for choice in choices] == ["default", "Alpha", "zeta"]
