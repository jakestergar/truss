"""Command-level tests for `truss train` subcommands in `truss.cli.train_commands`."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest
import rich_click as click
from click.testing import CliRunner

from truss.cli import train_commands
from truss.cli.cli import truss_cli
from truss.remote.baseten.remote import BasetenRemote

REMOTE_ARGS = ["--remote", "test_remote"]


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_remote():
    remote = Mock(spec=BasetenRemote)
    remote.api = Mock()
    remote.remote_url = "https://app.baseten.co"
    return remote


@pytest.fixture
def remote_factory(mock_remote):
    with patch(
        "truss.cli.train_commands.RemoteFactory.create", return_value=mock_remote
    ) as factory:
        yield factory


@pytest.fixture
def mock_log_watcher():
    with patch("truss.cli.train_commands.TrainingLogWatcher") as watcher_cls:
        watcher_cls.return_value.watch.return_value = [{"msg": "hello"}]
        yield watcher_cls


def _job_resp(status=None, job_object=None):
    resp = {"id": "job123", "training_project": {"id": "proj123", "name": "my-project"}}
    if status:
        resp["current_status"] = status
    if job_object:
        resp["job_object"] = job_object
    return resp


class TestPostCreateLogic:
    @pytest.mark.parametrize(
        "status, expected",
        [("TRAINING_JOB_PENDING", "pending"), ("TRAINING_JOB_QUEUED", "queued")],
    )
    def test_non_running_statuses_print_hint(self, mock_remote, status, expected):
        with patch("truss.cli.train_commands.console") as console:
            train_commands._handle_post_create_logic(
                _job_resp(status), mock_remote, tail=False
            )
        assert expected in console.print.call_args[0][0]

    def test_success_message_includes_cache_hint_when_cache_enabled(self, mock_remote):
        job_object = Mock()
        job_object.runtime.enable_cache = True
        with patch("truss.cli.train_commands.console") as console:
            train_commands._handle_post_create_logic(
                _job_resp(job_object=job_object), mock_remote, tail=False
            )
        printed = " ".join(str(c[0][0]) for c in console.print.call_args_list)
        assert "cache summarize" in printed

    def test_success_message_omits_cache_hint_without_job_object(self, mock_remote):
        with patch("truss.cli.train_commands.console") as console:
            train_commands._handle_post_create_logic(
                _job_resp(), mock_remote, tail=False
            )
        printed = " ".join(str(c[0][0]) for c in console.print.call_args_list)
        assert "cache summarize" not in printed
        assert "successfully created" in printed

    def test_tail_streams_logs(self, mock_remote, mock_log_watcher):
        with patch("truss.cli.train_commands.cli_log_utils.output_log") as output_log:
            train_commands._handle_post_create_logic(
                _job_resp(), mock_remote, tail=True
            )
        mock_log_watcher.assert_called_once_with(mock_remote.api, "proj123", "job123")
        output_log.assert_called_once_with({"msg": "hello"})


def test_prepare_click_context_inherits_root_obj():
    with click.Context(train_commands.train, obj={"log": "humanfriendly"}):
        ctx = train_commands._prepare_click_context(
            train_commands.stop_job, {"job_id": "job123"}
        )
    assert ctx.obj == {"log": "humanfriendly"}
    assert ctx.params == {"job_id": "job123"}


class TestFormatLocalTime:
    def test_empty_returns_empty(self):
        assert train_commands._format_local_time("") == ""

    def test_invalid_returns_input(self):
        assert train_commands._format_local_time("not-a-time") == "not-a-time"

    def test_valid_timestamp_is_formatted(self):
        assert train_commands._format_local_time("2024-01-01T12:00:00Z").count(":") >= 2


class TestDisplayISession:
    def _response(self, auth_codes):
        return {"auth_codes": auth_codes}

    def test_no_auth_codes_prints_nothing(self, mock_remote):
        mock_remote.api.get_training_job_isession.return_value = self._response([])
        with patch("truss.cli.train_commands.console") as console:
            train_commands._display_isession(mock_remote, "proj123", "job123")
        console.print.assert_not_called()

    def test_renders_table_sorted_by_replica_index(self, mock_remote):
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        mock_remote.api.get_training_job_isession.return_value = self._response(
            [
                {"replica_id": "job123r2", "auth_code": "b"},
                {"replica_id": "job123r1", "auth_code": "a"},
                {
                    "replica_id": "no-index",
                    "auth_code": "c",
                    "expires_at": expires_at,
                    "working_directory": "/workspace",
                    "generated_at": "2024-01-01T12:00:00Z",
                },
            ]
        )
        with patch("truss.cli.train_commands.console") as console:
            train_commands._display_isession(mock_remote, "proj123", "job123")

        table = console.print.call_args[0][0]
        assert [col.header for col in table.columns][-2:] == [
            "Expires In",
            "Working Directory",
        ]
        assert table.row_count == 3

    def test_unparseable_replica_id_does_not_raise(self, mock_remote):
        mock_remote.api.get_training_job_isession.return_value = self._response(
            [{"replica_id": "rabc", "auth_code": "a"}]
        )
        with patch("truss.cli.train_commands.console") as console:
            train_commands._display_isession(mock_remote, "proj123", "job123")
        assert console.print.called

    def test_api_errors_are_swallowed(self, mock_remote):
        mock_remote.api.get_training_job_isession.side_effect = Exception("boom")
        train_commands._display_isession(mock_remote, "proj123", "job123")


class TestResolveProjectId:
    def test_returns_none_without_identifiers(self, mock_remote):
        assert (
            train_commands._maybe_resolve_project_id_from_id_or_name(
                mock_remote, None, None
            )
            is None
        )

    def test_project_name_wins_over_project_id(self, mock_remote):
        with patch(
            "truss.cli.train_commands.train_cli.fetch_project_by_name_or_id",
            return_value={"id": "resolved"},
        ) as fetch:
            result = train_commands._maybe_resolve_project_id_from_id_or_name(
                mock_remote, project_id="proj123", project="my-project"
            )
        assert result == "resolved"
        assert fetch.call_args[0][1] == "my-project"


class TestRecreate:
    def test_recreate_invokes_core_and_post_create(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.recreate_training_job",
            return_value=_job_resp(),
        ) as recreate:
            result = runner.invoke(truss_cli, ["train", "recreate", *REMOTE_ARGS])

        assert result.exit_code == 0, result.output
        assert recreate.call_args[1]["job_id"] is None
        assert "successfully created" in result.output

    def test_recreate_prompts_for_remote_when_missing(self, runner, remote_factory):
        with (
            patch(
                "truss.cli.train_commands.remote_cli.inquire_remote_name",
                return_value="prompted_remote",
            ),
            patch(
                "truss.cli.train_commands.train_cli.recreate_training_job",
                return_value=_job_resp(),
            ),
        ):
            result = runner.invoke(truss_cli, ["train", "recreate"])

        assert result.exit_code == 0, result.output
        assert remote_factory.call_args[1]["remote"] == "prompted_remote"


class TestLogs:
    def test_non_tail_prints_paginated_logs(self, runner, remote_factory):
        with (
            patch(
                "truss.cli.train_commands.train_common.get_most_recent_job",
                return_value=("proj123", "job123"),
            ),
            patch("truss.cli.train_commands._display_isession"),
            patch(
                "truss.cli.train_commands.get_training_job_logs_with_pagination",
                return_value=["raw"],
            ) as get_logs,
            patch(
                "truss.cli.train_commands.cli_log_utils.parse_logs",
                return_value=[{"msg": "line"}],
            ),
            patch("truss.cli.train_commands.cli_log_utils.output_log") as output_log,
        ):
            result = runner.invoke(
                truss_cli, ["train", "logs", "--job-id", "job123", *REMOTE_ARGS]
            )

        assert result.exit_code == 0, result.output
        get_logs.assert_called_once()
        output_log.assert_called_once_with({"msg": "line"})

    def test_tail_streams_logs(self, runner, remote_factory, mock_log_watcher):
        with (
            patch(
                "truss.cli.train_commands.train_common.get_most_recent_job",
                return_value=("proj123", "job123"),
            ),
            patch("truss.cli.train_commands._display_isession"),
            patch("truss.cli.train_commands.cli_log_utils.output_log") as output_log,
        ):
            result = runner.invoke(truss_cli, ["train", "logs", "--tail", *REMOTE_ARGS])

        assert result.exit_code == 0, result.output
        output_log.assert_called_once_with({"msg": "hello"})


class TestStop:
    def test_stop_all(self, runner, remote_factory, mock_remote):
        with (
            patch(
                "truss.cli.train_commands.train_cli.fetch_project_by_name_or_id",
                return_value={"id": "proj123"},
            ),
            patch("truss.cli.train_commands.train_cli.stop_all_jobs") as stop_all,
        ):
            result = runner.invoke(
                truss_cli,
                ["train", "stop", "--all", "--project", "my-project", *REMOTE_ARGS],
            )

        assert result.exit_code == 0, result.output
        stop_all.assert_called_once_with(mock_remote, "proj123")
        mock_remote.api.stop_training_job.assert_not_called()

    def test_stop_single_job(self, runner, remote_factory, mock_remote):
        with patch(
            "truss.cli.train_commands.train_cli.get_args_for_stop",
            return_value=("proj123", "job123"),
        ):
            result = runner.invoke(
                truss_cli, ["train", "stop", "--job-id", "job123", *REMOTE_ARGS]
            )

        assert result.exit_code == 0, result.output
        mock_remote.api.stop_training_job.assert_called_once_with("proj123", "job123")
        assert "stopped successfully" in result.output


def test_view_training(runner, remote_factory, mock_remote):
    with patch(
        "truss.cli.train_commands.train_cli.view_training_details"
    ) as view_details:
        result = runner.invoke(
            truss_cli, ["train", "view", "--job-id", "job123", *REMOTE_ARGS]
        )

    assert result.exit_code == 0, result.output
    view_details.assert_called_once_with(mock_remote, None, "job123")


def test_view_training_warns_when_both_project_flags_given(
    runner, remote_factory, mock_remote
):
    with (
        patch(
            "truss.cli.train_commands.train_cli.fetch_project_by_name_or_id",
            return_value={"id": "proj123"},
        ),
        patch("truss.cli.train_commands.train_cli.view_training_details"),
    ):
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "view",
                "--project-id",
                "proj123",
                "--project",
                "my-project",
                *REMOTE_ARGS,
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Using `project`" in result.output


def test_metrics(runner, remote_factory, mock_remote):
    with patch(
        "truss.cli.train_commands.train_cli.view_training_job_metrics"
    ) as metrics:
        result = runner.invoke(
            truss_cli, ["train", "metrics", "--job-id", "job123", *REMOTE_ARGS]
        )

    assert result.exit_code == 0, result.output
    metrics.assert_called_once_with(mock_remote, None, "job123")


class TestDeployCheckpoints:
    def _patches(self):
        return (
            patch(
                "truss.cli.train_commands.train_cli.create_model_version_from_inference_template"
            ),
            patch("truss.cli.train_commands.train_cli.write_truss_config"),
            patch(
                "truss.cli.train_commands.train_cli.print_deploy_checkpoints_success_message"
            ),
        )

    def test_deploy(self, runner, remote_factory):
        create, write_config, success_message = self._patches()
        with create as create_mock, write_config as write_mock, success_message as msg:
            result = runner.invoke(
                truss_cli,
                ["train", "deploy_checkpoints", "--job-id", "job123", *REMOTE_ARGS],
            )

        assert result.exit_code == 0, result.output
        write_mock.assert_called_once_with(create_mock.return_value, None, False)
        msg.assert_called_once_with(create_mock.return_value.deploy_config)

    def test_dry_run_skips_deploy_message(self, runner, remote_factory):
        create, write_config, success_message = self._patches()
        with create, write_config as write_mock, success_message as msg:
            result = runner.invoke(
                truss_cli,
                [
                    "train",
                    "deploy_checkpoints",
                    "--job-id",
                    "job123",
                    "--dry-run",
                    "--truss-config-output-dir",
                    "out",
                    *REMOTE_ARGS,
                ],
            )

        assert result.exit_code == 0, result.output
        assert "did not deploy" in result.output
        assert write_mock.call_args[0][1:] == ("out", True)
        msg.assert_not_called()


class TestDownload:
    def test_download_success(self, runner, remote_factory, mock_remote):
        with patch(
            "truss.cli.train_commands.train_cli.download_training_job_data",
            return_value="/tmp/out.zip",
        ) as download:
            result = runner.invoke(
                truss_cli,
                ["train", "download", "--job-id", "job123", "--no-unzip", *REMOTE_ARGS],
            )

        assert result.exit_code == 0, result.output
        assert download.call_args[1]["unzip"] is False
        assert "/tmp/out.zip" in result.output

    def test_download_failure_exits_nonzero(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.download_training_job_data",
            side_effect=Exception("nope"),
        ):
            result = runner.invoke(
                truss_cli, ["train", "download", "--job-id", "job123", *REMOTE_ARGS]
            )

        assert result.exit_code == 1
        assert "Failed to download training job data" in result.output

    def test_checkpoint_artifacts_success(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.download_checkpoint_artifacts",
            return_value="/tmp/checkpoints",
        ):
            result = runner.invoke(
                truss_cli,
                ["train", "get_checkpoint_urls", "--job-id", "job123", *REMOTE_ARGS],
            )

        assert result.exit_code == 0, result.output
        assert "/tmp/checkpoints" in result.output

    def test_checkpoint_artifacts_failure_exits_nonzero(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.download_checkpoint_artifacts",
            side_effect=Exception("nope"),
        ):
            result = runner.invoke(
                truss_cli, ["train", "get_checkpoint_urls", *REMOTE_ARGS]
            )

        assert result.exit_code == 1
        assert "Failed to download checkpoint artifacts" in result.output


class TestInit:
    def test_list_examples(self, runner):
        with patch(
            "truss.cli.train_commands.train_cli._get_all_train_init_example_options",
            return_value=["sft", "grpo"],
        ):
            result = runner.invoke(truss_cli, ["train", "init", "--list-examples"])

        assert result.exit_code == 0, result.output
        assert "- sft" in result.output
        assert "- grpo" in result.output

    def test_empty_project_scaffold(self, runner):
        with runner.isolated_filesystem():
            result = runner.invoke(truss_cli, ["train", "init"])
            assert result.exit_code == 0, result.output
            assert "truss-train-init" in result.output

    def test_downloads_selected_example(self, runner):
        with (
            runner.isolated_filesystem(),
            patch(
                "truss.cli.train_commands.train_cli._get_train_init_example_info",
                return_value=[{"url": "https://api.github.com/sft"}],
            ),
            patch(
                "truss.cli.train_commands.train_cli.download_git_directory",
                return_value=True,
            ) as download,
        ):
            result = runner.invoke(truss_cli, ["train", "init", "--examples", "sft"])

            assert result.exit_code == 0, result.output
            assert download.call_args[1]["git_api_url"] == "https://api.github.com/sft"
            assert "initialized at" in result.output

    def test_unknown_example_is_reported_and_skipped(self, runner):
        with (
            runner.isolated_filesystem(),
            patch(
                "truss.cli.train_commands.train_cli._get_train_init_example_info",
                return_value=None,
            ),
            patch(
                "truss.cli.train_commands.train_cli._get_all_train_init_example_options",
                return_value=["sft"],
            ),
            patch(
                "truss.cli.train_commands.train_cli.download_git_directory"
            ) as download,
        ):
            result = runner.invoke(truss_cli, ["train", "init", "--examples", "bogus"])

            assert result.exit_code == 0, result.output
            assert "not found in the ml-cookbook repository" in result.output
            download.assert_not_called()

    def test_failed_download_is_reported(self, runner):
        with (
            runner.isolated_filesystem(),
            patch(
                "truss.cli.train_commands.train_cli._get_train_init_example_info",
                return_value=[{"url": "https://api.github.com/sft"}],
            ),
            patch(
                "truss.cli.train_commands.train_cli.download_git_directory",
                return_value=False,
            ),
        ):
            result = runner.invoke(truss_cli, ["train", "init", "--examples", "sft"])

            assert result.exit_code == 0, result.output
            assert "Failed to initialize training artifacts to" in result.output

    def test_unexpected_error_exits_nonzero(self, runner):
        with patch(
            "truss.cli.train_commands.train_cli._get_all_train_init_example_options",
            side_effect=Exception("boom"),
        ):
            result = runner.invoke(truss_cli, ["train", "init", "--list-examples"])

        assert result.exit_code == 1
        assert "Failed to initialize training artifacts" in result.output


def test_cache_summarize(runner, remote_factory, mock_remote):
    with patch(
        "truss.cli.train_commands.train_cli.view_cache_summary_by_project"
    ) as summarize:
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "cache",
                "summarize",
                "my-project",
                "--sort",
                "size",
                "--order",
                "desc",
                *REMOTE_ARGS,
            ],
        )

    assert result.exit_code == 0, result.output
    summarize.assert_called_once_with(
        mock_remote, "my-project", "size", "desc", "cli-table"
    )


def test_checkpoints_list(runner, remote_factory, mock_remote):
    with (
        patch(
            "truss.cli.train_commands.train_common.get_most_recent_job",
            return_value=("proj123", "job123"),
        ),
        patch(
            "truss.cli.train_commands.common.check_is_interactive", return_value=False
        ),
        patch(
            "truss.cli.train_commands.checkpoint_mod.view_checkpoint_list"
        ) as view_list,
    ):
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "checkpoints",
                "list",
                "--job-id",
                "job123",
                "--checkpoint-name",
                "ckpt-1",
                *REMOTE_ARGS,
            ],
        )

    assert result.exit_code == 0, result.output
    assert view_list.call_args[1]["checkpoint_name"] == "ckpt-1"
    assert view_list.call_args[1]["interactive"] is False


class TestUpdateSession:
    def test_requires_at_least_one_field(self, runner):
        result = runner.invoke(truss_cli, ["train", "update_session", "job123"])
        assert result.exit_code == 1
        assert "At least one of --trigger or --timeout-minutes" in result.output

    def test_unknown_job_exits_nonzero(self, runner, remote_factory, mock_remote):
        mock_remote.api.search_training_jobs.return_value = []
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "update_session",
                "job123",
                "--trigger",
                "on_demand",
                *REMOTE_ARGS,
            ],
        )
        assert result.exit_code == 1
        assert "No training job found with ID: job123" in result.output

    def test_updates_session(self, runner, remote_factory, mock_remote):
        mock_remote.api.search_training_jobs.return_value = [
            {"training_project": {"id": "proj123"}}
        ]
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "update_session",
                "job123",
                "--timeout-minutes",
                "30",
                *REMOTE_ARGS,
            ],
        )

        assert result.exit_code == 0, result.output
        mock_remote.api.update_interactive_session.assert_called_once_with(
            project_id="proj123", job_id="job123", trigger=None, timeout_minutes=30
        )

    def test_api_failure_exits_nonzero(self, runner, remote_factory, mock_remote):
        mock_remote.api.search_training_jobs.return_value = [
            {"training_project": {"id": "proj123"}}
        ]
        mock_remote.api.update_interactive_session.side_effect = Exception("boom")
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "update_session",
                "job123",
                "--timeout-minutes",
                "30",
                *REMOTE_ARGS,
            ],
        )

        assert result.exit_code == 1
        assert "Failed to update interactive session" in result.output


class TestUpdate:
    def test_requires_a_field(self, runner):
        result = runner.invoke(
            truss_cli, ["train", "update", "--job-id", "job123", *REMOTE_ARGS]
        )
        assert result.exit_code == 2
        assert "At least one field to update must be provided." in result.output

    def test_updates_priority(self, runner, remote_factory, mock_remote):
        with patch(
            "truss.cli.train_commands.train_cli.update_training_job",
            return_value={"id": "job123"},
        ) as update_job:
            result = runner.invoke(
                truss_cli,
                [
                    "train",
                    "update",
                    "--job-id",
                    "job123",
                    "--priority",
                    "5",
                    *REMOTE_ARGS,
                ],
            )

        assert result.exit_code == 0, result.output
        assert update_job.call_args[1]["priority"] == 5
        assert "Training job job123 updated." in result.output

    def test_failure_exits_nonzero(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.update_training_job",
            side_effect=Exception("boom"),
        ):
            result = runner.invoke(
                truss_cli,
                [
                    "train",
                    "update",
                    "--job-id",
                    "job123",
                    "--priority",
                    "5",
                    *REMOTE_ARGS,
                ],
            )

        assert result.exit_code == 1
        assert "Failed to update training job" in result.output


class TestGetISession:
    def _setup_job(self, mock_remote, auth_codes):
        mock_remote.api.search_training_jobs.return_value = [
            {"training_project": {"id": "proj123"}}
        ]
        mock_remote.api.get_training_job_isession.return_value = {
            "auth_codes": auth_codes
        }

    def test_unknown_job_exits_nonzero(self, runner, remote_factory, mock_remote):
        mock_remote.api.search_training_jobs.return_value = []
        result = runner.invoke(
            truss_cli, ["train", "isession", "--job-id", "job123", *REMOTE_ARGS]
        )
        assert result.exit_code == 1
        assert "No training job found with ID: job123" in result.output

    def test_no_auth_codes(self, runner, remote_factory, mock_remote):
        self._setup_job(mock_remote, [])
        result = runner.invoke(
            truss_cli, ["train", "isession", "--job-id", "job123", *REMOTE_ARGS]
        )
        assert result.exit_code == 0, result.output
        assert "No auth codes found" in result.output

    def test_table_output(self, runner, remote_factory, mock_remote):
        self._setup_job(mock_remote, [{"replica_id": "r0", "auth_code": "abc"}])
        result = runner.invoke(
            truss_cli, ["train", "isession", "--job-id", "job123", *REMOTE_ARGS]
        )
        assert result.exit_code == 0, result.output
        assert "abc" in result.output

    def test_on_startup_trigger_cannot_be_changed(
        self, runner, remote_factory, mock_remote
    ):
        self._setup_job(mock_remote, [{"session_id": "s1", "trigger": "on_startup"}])
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "isession",
                "--job-id",
                "job123",
                "--update-trigger",
                "on_demand",
                *REMOTE_ARGS,
            ],
        )
        assert result.exit_code == 1
        assert "Cannot change trigger on on_startup sessions" in result.output

    def test_json_output_includes_patch_messages(
        self, runner, remote_factory, mock_remote
    ):
        self._setup_job(mock_remote, [{"session_id": "s1", "replica_id": "r0"}])
        mock_remote.api.patch_interactive_session.return_value = {"message": "extended"}

        result = runner.invoke(
            truss_cli,
            [
                "train",
                "isession",
                "--job-id",
                "job123",
                "--update-timeout",
                "60",
                "--format",
                "json",
                *REMOTE_ARGS,
            ],
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["update_messages"] == ["extended"]

    def test_api_failure_exits_nonzero(self, runner, remote_factory, mock_remote):
        mock_remote.api.search_training_jobs.return_value = [
            {"training_project": {"id": "proj123"}}
        ]
        mock_remote.api.get_training_job_isession.side_effect = Exception("boom")
        result = runner.invoke(
            truss_cli, ["train", "isession", "--job-id", "job123", *REMOTE_ARGS]
        )
        assert result.exit_code == 1
        assert "Failed to get auth codes" in result.output


class TestPatchSessions:
    def test_skips_sessions_without_id_and_reports_updates(self, mock_remote):
        mock_remote.api.patch_interactive_session.return_value = {"message": "ok"}
        with patch("truss.cli.train_commands.console") as console:
            messages = train_commands._patch_sessions(
                mock_remote,
                "proj123",
                "job123",
                [{"replica_id": "r0"}, {"session_id": "s1", "replica_id": "r1"}],
                timeout_minutes=15,
            )

        assert messages == ["ok"]
        printed = " ".join(str(c[0][0]) for c in console.print.call_args_list)
        assert "No session_id found for replica r0" in printed
        assert "Successfully updated 1 session(s)" in printed

    def test_quiet_mode_collects_without_printing(self, mock_remote):
        mock_remote.api.patch_interactive_session.return_value = {"message": "ok"}
        with patch("truss.cli.train_commands.console") as console:
            messages = train_commands._patch_sessions(
                mock_remote,
                "proj123",
                "job123",
                [{"replica_id": "r0"}, {"session_id": "s1"}],
                trigger="on_demand",
                quiet=True,
            )

        assert messages == ["ok"]
        console.print.assert_not_called()

    def test_all_failures_exit_nonzero(self, mock_remote):
        mock_remote.api.patch_interactive_session.side_effect = Exception("boom")
        with pytest.raises(SystemExit):
            train_commands._patch_sessions(
                mock_remote, "proj123", "job123", [{"session_id": "s1"}]
            )


class TestCapacity:
    def test_view(self, runner, remote_factory, mock_remote):
        with patch(
            "truss.cli.train_commands.train_cli.display_training_capacity"
        ) as display:
            result = runner.invoke(
                truss_cli, ["train", "capacity", "view", *REMOTE_ARGS]
            )

        assert result.exit_code == 0, result.output
        display.assert_called_once_with(mock_remote)

    def test_update(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.update_team_training_gpu_capacity",
            return_value={"team_name": "team-a", "gpu_type": "H100", "limit": 8},
        ):
            result = runner.invoke(
                truss_cli,
                [
                    "train",
                    "capacity",
                    "update",
                    "--team",
                    "team-a",
                    "--gpu-type",
                    "H100",
                    "--capacity",
                    "8",
                    *REMOTE_ARGS,
                ],
            )

        assert result.exit_code == 0, result.output
        assert "capacity to 8" in result.output

    def test_update_click_exception_propagates(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.update_team_training_gpu_capacity",
            side_effect=click.ClickException("not an admin"),
        ):
            result = runner.invoke(
                truss_cli,
                [
                    "train",
                    "capacity",
                    "update",
                    "--team",
                    "team-a",
                    "--gpu-type",
                    "H100",
                    "--capacity",
                    "8",
                    *REMOTE_ARGS,
                ],
            )

        assert result.exit_code == 1
        assert "not an admin" in result.output

    def test_update_failure_exits_nonzero(self, runner, remote_factory):
        with patch(
            "truss.cli.train_commands.train_cli.update_team_training_gpu_capacity",
            side_effect=Exception("boom"),
        ):
            result = runner.invoke(
                truss_cli,
                [
                    "train",
                    "capacity",
                    "update",
                    "--team",
                    "team-a",
                    "--gpu-type",
                    "H100",
                    "--capacity",
                    "8",
                    *REMOTE_ARGS,
                ],
            )

        assert result.exit_code == 1
        assert "Failed to update team capacity" in result.output


class TestWorkstation:
    @pytest.fixture
    def workstation_env(self, remote_factory):
        with (
            patch(
                "truss.cli.train_commands.RemoteFactory.get_remote_team",
                return_value=None,
            ),
            patch(
                "truss.cli.train_commands._resolve_team_name",
                return_value=("team-a", "team1"),
            ),
            patch("truss_train.public_api.push") as push,
        ):
            push.return_value = _job_resp()
            yield push

    def test_gpu_count_and_node_count_are_mutually_exclusive(self, runner):
        result = runner.invoke(
            truss_cli, ["train", "workstation", "--gpu-count", "2", "--node-count", "2"]
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_multi_node_copies_templates_and_lists_hosts(self, runner, workstation_env):
        with patch(
            "truss.cli.train_commands.copy_workstation_templates"
        ) as copy_templates:
            result = runner.invoke(
                truss_cli, ["train", "workstation", "--node-count", "2", *REMOTE_ARGS]
            )

        assert result.exit_code == 0, result.output
        copy_templates.assert_called_once()
        assert "(leader)" in result.output
        assert "training-job-job123-1.ssh.baseten.co" in result.output
        assert "BT_LEADER_ADDR" in result.output

    def test_single_node_defaults(self, runner, workstation_env):
        result = runner.invoke(truss_cli, ["train", "workstation", *REMOTE_ARGS])

        assert result.exit_code == 0, result.output
        assert "BT_LEADER_ADDR" not in result.output
        project = workstation_env.call_args[1]["config"]
        assert project.name == "workstation-H100"
        assert project.job.compute.accelerator.count == 1

    def test_tail_streams_logs(self, runner, workstation_env, mock_log_watcher):
        with patch("truss.cli.train_commands.cli_log_utils.output_log") as output_log:
            result = runner.invoke(
                truss_cli, ["train", "workstation", "--tail", *REMOTE_ARGS]
            )

        assert result.exit_code == 0, result.output
        output_log.assert_called_once_with({"msg": "hello"})
