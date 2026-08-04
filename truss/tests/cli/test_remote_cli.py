"""Tests for `truss.cli.remote_cli` prompt helpers.

All InquirerPy prompts are patched at the module boundary, so nothing here
requires a TTY, network access or a `.trussrc` file on disk.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import pytest
from InquirerPy.validator import ValidationError

from truss.base.constants import DEFAULT_REMOTE_NAME, DEFAULT_REMOTE_URL
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
from truss.remote.baseten.oauth import OAuthCredential, OAuthError
from truss.remote.remote_factory import AuthType


def _document(text):
    return SimpleNamespace(text=text)


class TestNonEmptyValidator:
    @pytest.mark.parametrize("text", ["", "   ", "\t\n"])
    def test_rejects_blank_input(self, text):
        with pytest.raises(ValidationError) as exc_info:
            NonEmptyValidator().validate(_document(text))

        assert "non-empty" in str(exc_info.value)
        assert exc_info.value.cursor_position == len(text)

    @pytest.mark.parametrize("text", ["abc", "  padded  "])
    def test_accepts_non_blank_input(self, text):
        assert NonEmptyValidator().validate(_document(text)) is None


class TestInquireRemoteConfig:
    def test_api_key_flow_uses_defaults(self):
        with (
            patch("truss.cli.remote_cli.inquirer.select") as mock_select,
            patch("truss.cli.remote_cli.inquirer.secret") as mock_secret,
        ):
            mock_select.return_value.execute.return_value = "api_key"
            mock_secret.return_value.execute.return_value = "secret-key"

            remote_config = inquire_remote_config()

        assert remote_config.name == DEFAULT_REMOTE_NAME
        assert remote_config.configs == {
            "remote_provider": DEFAULT_REMOTE_NAME,
            "auth_type": AuthType.API_KEY,
            "api_key": "secret-key",
            "remote_url": DEFAULT_REMOTE_URL,
        }
        # The API key prompt must be validated as non-empty.
        assert isinstance(mock_secret.call_args.kwargs["validate"], NonEmptyValidator)

    def test_api_key_flow_honors_custom_name_and_url(self):
        with (
            patch("truss.cli.remote_cli.inquirer.select") as mock_select,
            patch("truss.cli.remote_cli.inquirer.secret") as mock_secret,
        ):
            mock_select.return_value.execute.return_value = "api_key"
            mock_secret.return_value.execute.return_value = "k"

            remote_config = inquire_remote_config(
                remote_name="staging", remote_url="https://app.staging.baseten.co"
            )

        assert remote_config.name == "staging"
        assert remote_config.configs["remote_url"] == "https://app.staging.baseten.co"

    def test_browser_flow_returns_oauth_config(self):
        credential = OAuthCredential(
            access_token="atk", refresh_token="rtk", expires_at=123
        )
        with (
            patch("truss.cli.remote_cli.inquirer.select") as mock_select,
            patch(
                "truss.cli.remote_cli.oauth.run_device_flow", return_value=credential
            ) as mock_flow,
            patch("truss.cli.remote_cli.inquirer.secret") as mock_secret,
        ):
            mock_select.return_value.execute.return_value = "browser"

            remote_config = inquire_remote_config()

        mock_flow.assert_called_once_with("https://api.baseten.co")
        mock_secret.assert_not_called()
        assert remote_config.name == DEFAULT_REMOTE_NAME
        assert remote_config.configs == {
            "remote_provider": DEFAULT_REMOTE_NAME,
            "remote_url": DEFAULT_REMOTE_URL,
            "auth_type": AuthType.OAUTH,
            "oauth_access_token": "atk",
            "oauth_refresh_token": "rtk",
            "oauth_expires_at": "123",
        }

    def test_browser_flow_resolves_rest_api_url_for_custom_remote(self):
        credential = OAuthCredential(access_token="a", refresh_token="r", expires_at=1)
        with (
            patch("truss.cli.remote_cli.inquirer.select") as mock_select,
            patch(
                "truss.cli.remote_cli.resolve_rest_api_url",
                return_value="https://api.example.com",
            ) as mock_resolve,
            patch(
                "truss.cli.remote_cli.oauth.run_device_flow", return_value=credential
            ) as mock_flow,
        ):
            mock_select.return_value.execute.return_value = "browser"

            remote_config = inquire_remote_config(
                remote_name="custom", remote_url="https://app.example.com"
            )

        mock_resolve.assert_called_once_with("https://app.example.com")
        mock_flow.assert_called_once_with("https://api.example.com")
        assert remote_config.configs["remote_url"] == "https://app.example.com"

    def test_browser_flow_converts_oauth_error_to_click_exception(self):
        with (
            patch("truss.cli.remote_cli.inquirer.select") as mock_select,
            patch(
                "truss.cli.remote_cli.oauth.run_device_flow",
                side_effect=OAuthError("device flow denied"),
            ),
        ):
            mock_select.return_value.execute.return_value = "browser"

            with pytest.raises(click.ClickException) as exc_info:
                inquire_remote_config()

        assert "device flow denied" in str(exc_info.value)


class TestInquireRemoteName:
    @patch("truss.cli.remote_cli.RemoteFactory.get_available_config_names")
    def test_no_remotes_without_create_raises(self, mock_names):
        mock_names.return_value = []

        with pytest.raises(click.ClickException) as exc_info:
            inquire_remote_name(allow_create=False)

        assert "truss auth login" in str(exc_info.value)

    @patch("truss.cli.remote_cli.check_is_interactive", return_value=False)
    @patch("truss.cli.remote_cli.RemoteFactory.get_available_config_names")
    def test_no_remotes_non_interactive_raises_usage_error(
        self, mock_names, mock_interactive
    ):
        mock_names.return_value = []

        with pytest.raises(click.UsageError) as exc_info:
            inquire_remote_name()

        assert "truss login" in str(exc_info.value)

    @patch("truss.cli.remote_cli.RemoteFactory.update_remote_config")
    @patch("truss.cli.remote_cli.inquire_remote_config")
    @patch("truss.cli.remote_cli.check_is_interactive", return_value=True)
    @patch("truss.cli.remote_cli.RemoteFactory.get_available_config_names")
    def test_no_remotes_interactive_creates_and_saves_config(
        self, mock_names, mock_interactive, mock_inquire_config, mock_update, capsys
    ):
        mock_names.return_value = []
        new_config = MagicMock()
        new_config.name = "baseten"
        mock_inquire_config.return_value = new_config

        assert inquire_remote_name() == "baseten"

        mock_update.assert_called_once_with(new_config)
        assert "baseten" in capsys.readouterr().out

    @patch("truss.cli.remote_cli.RemoteFactory.get_available_config_names")
    def test_single_remote_is_returned_without_prompting(self, mock_names):
        mock_names.return_value = ["only-one"]

        with patch("truss.cli.remote_cli.inquirer.select") as mock_select:
            assert inquire_remote_name() == "only-one"

        mock_select.assert_not_called()

    @patch("truss.cli.remote_cli.check_is_interactive", return_value=False)
    @patch("truss.cli.remote_cli.RemoteFactory.get_available_config_names")
    def test_multiple_remotes_non_interactive_raises_usage_error(
        self, mock_names, mock_interactive
    ):
        mock_names.return_value = ["a", "b"]

        with pytest.raises(click.UsageError) as exc_info:
            inquire_remote_name()

        assert "--remote" in str(exc_info.value)

    @patch("truss.cli.remote_cli.check_is_interactive", return_value=True)
    @patch("truss.cli.remote_cli.RemoteFactory.get_available_config_names")
    def test_multiple_remotes_interactive_prompts_with_all_choices(
        self, mock_names, mock_interactive
    ):
        mock_names.return_value = ["a", "b"]

        with patch("truss.cli.remote_cli.inquirer.select") as mock_select:
            mock_select.return_value.execute.return_value = "b"

            assert inquire_remote_name() == "b"

        assert mock_select.call_args.kwargs["choices"] == ["a", "b"]


def test_inquire_model_name_returns_prompt_result():
    with patch("truss.cli.remote_cli.inquirer.text") as mock_text:
        mock_text.return_value.execute.return_value = "my-model"

        assert inquire_model_name() == "my-model"

    assert mock_text.call_args.kwargs["qmark"] == ""


class TestTeamHelpers:
    @pytest.mark.parametrize(
        "team_name,expected",
        [
            ("Team Alpha", "team1"),
            ("Team Beta", "team2"),
            ("Unknown", None),
            ("", None),
        ],
    )
    def test_get_team_id_from_name(self, team_name, expected):
        teams = {
            "Team Alpha": TeamType(id="team1", name="Team Alpha", default=True),
            "Team Beta": TeamType(id="team2", name="Team Beta", default=False),
        }

        assert get_team_id_from_name(teams, team_name) == expected

    def test_get_team_id_from_name_with_no_teams(self):
        assert get_team_id_from_name({}, "Team Alpha") is None

    def test_format_available_teams(self):
        teams = {
            "Team Alpha": TeamType(id="team1", name="Team Alpha", default=True),
            "Team Beta": TeamType(id="team2", name="Team Beta", default=False),
        }

        assert format_available_teams(teams) == "Team Alpha, Team Beta"

    def test_format_available_teams_when_empty(self):
        assert format_available_teams({}) == "none"


class TestInquireTeam:
    def test_returns_none_when_no_existing_teams_passed(self):
        with patch("truss.cli.remote_cli.inquirer.select") as mock_select:
            assert inquire_team() is None

        mock_select.assert_not_called()

    @patch("truss.cli.remote_cli.check_is_interactive", return_value=True)
    def test_sorts_default_team_first_then_case_insensitive(self, mock_interactive):
        teams = {
            "zeta": TeamType(id="t1", name="zeta", default=False),
            "Alpha": TeamType(id="t2", name="Alpha", default=False),
            "middle": TeamType(id="t3", name="middle", default=True),
        }

        with patch("truss.cli.remote_cli.inquirer.select") as mock_select:
            mock_select.return_value.execute.return_value = "Alpha"

            assert inquire_team(existing_teams=teams) == "Alpha"

        choices = mock_select.call_args.kwargs["choices"]
        assert [choice.value for choice in choices] == ["middle", "Alpha", "zeta"]
        assert [choice.name for choice in choices] == [
            "middle (default)",
            "Alpha",
            "zeta",
        ]

    @patch("truss.cli.remote_cli.check_is_interactive", return_value=True)
    def test_custom_prompt_is_forwarded(self, mock_interactive):
        teams = {"Team Alpha": TeamType(id="t1", name="Team Alpha", default=True)}

        with patch("truss.cli.remote_cli.inquirer.select") as mock_select:
            mock_select.return_value.execute.return_value = "Team Alpha"

            inquire_team(existing_teams=teams, prompt="Pick a team:")

        assert mock_select.call_args.args[0] == "Pick a team:"

    @patch("truss.cli.remote_cli.check_is_interactive", return_value=False)
    def test_non_interactive_with_empty_teams_reports_none_available(
        self, mock_interactive
    ):
        with pytest.raises(click.UsageError) as exc_info:
            inquire_team(existing_teams={})

        assert "available: none" in str(exc_info.value)
