from unittest.mock import Mock, patch

import pytest

from truss.api import definitions, login, push, whoami


def test_model_deployment_repr():
    svc = Mock()
    svc.model_id = "model-1"
    svc.model_version_id = "deployment-1"
    deployment = definitions.ModelDeployment(svc)
    assert "model-1" in repr(deployment)
    assert "deployment-1" in repr(deployment)


def test_model_deployment_wait_for_active_success():
    svc = Mock()
    svc.poll_deployment_status.return_value = ["BUILDING", "ACTIVE"]
    deployment = definitions.ModelDeployment(svc)
    assert deployment.wait_for_active(timeout_seconds=600) is True


def test_model_deployment_wait_for_active_fails():
    svc = Mock()
    svc.poll_deployment_status.return_value = ["BUILDING", "FAILED"]
    deployment = definitions.ModelDeployment(svc)
    with pytest.raises(ValueError, match="Deployment failed"):
        deployment.wait_for_active(timeout_seconds=600)


def test_model_deployment_wait_for_active_timeout():
    svc = Mock()
    svc.poll_deployment_status.return_value = ["BUILDING"]
    deployment = definitions.ModelDeployment(svc)
    with pytest.raises(TimeoutError):
        deployment.wait_for_active(timeout_seconds=-1)


@patch("truss.api.RemoteFactory.update_remote_config")
def test_login(mock_update):
    login("my-api-key")
    mock_update.assert_called_once()


@patch("truss.api.RemoteFactory")
def test_whoami_single_remote(mock_remote_factory):
    mock_remote_factory.get_available_config_names.return_value = ["baseten"]
    mock_remote = Mock()
    mock_remote.whoami.return_value = {"email": "user@example.com"}
    mock_remote_factory.create.return_value = mock_remote
    assert whoami() == {"email": "user@example.com"}


@patch("truss.api.RemoteFactory")
def test_whoami_no_remote(mock_remote_factory):
    mock_remote_factory.get_available_config_names.return_value = []
    with pytest.raises(ValueError, match="Please authenticate"):
        whoami()


@patch("truss.api.RemoteFactory")
def test_whoami_multiple_remotes(mock_remote_factory):
    mock_remote_factory.get_available_config_names.return_value = ["a", "b"]
    with pytest.raises(ValueError, match="Multiple remotes"):
        whoami()


@patch("truss.api.load")
@patch("truss.api.RemoteFactory")
def test_push_with_invalid_labels(mock_remote_factory, mock_load, tmp_path):
    mock_tr = Mock()
    mock_tr.spec.config.model_name = "test-model"
    mock_load.return_value = mock_tr
    with pytest.raises(ValueError, match="labels must be a JSON-serializable"):
        push(str(tmp_path), labels="not-a-dict")


@patch("truss.api.load")
@patch("truss.api.RemoteFactory")
def test_push_happy_path(mock_remote_factory, mock_load, tmp_path):
    mock_tr = Mock()
    mock_tr.spec.config.model_name = "test-model"
    mock_load.return_value = mock_tr

    remote = Mock()
    remote.api.get_teams.return_value = {"my-team": Mock(id="team-1", name="my-team")}
    remote.api.models.return_value = {"models": []}
    remote.push.return_value = Mock(model_id="m", model_version_id="v")
    mock_remote_factory.get_available_config_names.return_value = ["baseten"]
    mock_remote_factory.create.return_value = remote

    result = push(str(tmp_path))
    assert isinstance(result, definitions.ModelDeployment)


@patch("truss.api.RemoteFactory")
def test_push_without_remote_or_auth_raises(mock_remote_factory, tmp_path):
    mock_remote_factory.get_available_config_names.return_value = []
    with pytest.raises(ValueError, match="Please authenticate"):
        push(str(tmp_path))
