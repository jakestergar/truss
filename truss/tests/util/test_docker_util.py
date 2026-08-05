from unittest import mock

import pytest
import python_on_whales
from python_on_whales.exceptions import DockerException

from truss.base.constants import TRUSS, TRUSS_DIR
from truss.local.local_config import LocalConfig
from truss.util import docker as docker_util


@pytest.fixture(autouse=True)
def reset_docker_client():
    original_client = docker_util.Docker._client
    docker_util.Docker._client = None
    try:
        yield
    finally:
        docker_util.Docker._client = original_client


@pytest.fixture
def client():
    fake_client = mock.MagicMock()
    docker_util.Docker._client = fake_client
    return fake_client


def test_client_without_sudo_is_default_client_and_is_cached():
    with mock.patch.object(
        docker_util.LocalConfigHandler, "get_config", return_value=LocalConfig()
    ) as get_config:
        assert docker_util.Docker.client() is python_on_whales.docker
        assert docker_util.Docker.client() is python_on_whales.docker

    get_config.assert_called_once()


def test_client_with_sudo_uses_sudo_docker_call():
    sudo_client = mock.MagicMock()
    with (
        mock.patch.object(
            docker_util.LocalConfigHandler,
            "get_config",
            return_value=LocalConfig(use_sudo=True),
        ),
        mock.patch(
            "python_on_whales.DockerClient", return_value=sudo_client
        ) as docker_client_cls,
    ):
        assert docker_util.Docker.client() is sudo_client
        assert docker_util.Docker.client() is sudo_client

    docker_client_cls.assert_called_once_with(client_call=["sudo", "docker"])


def test_get_containers_passes_label_filters(client):
    containers = docker_util.get_containers({TRUSS: True, TRUSS_DIR: "/some/dir"})

    assert containers is client.container.list.return_value
    client.container.list.assert_called_once_with(
        filters={"label=truss": True, "label=truss_dir": "/some/dir"}, all=False
    )


def test_get_containers_can_include_stopped_containers(client):
    docker_util.get_containers({TRUSS: True}, all=True)

    client.container.list.assert_called_once_with(
        filters={"label=truss": True}, all=True
    )


def test_get_images_passes_label_filters(client):
    images = docker_util.get_images({TRUSS: True})

    assert images is client.image.list.return_value
    client.image.list.assert_called_once_with(filters={"label=truss": True})


def _container_with_ports(ports):
    container = mock.MagicMock()
    container.network_settings.ports = ports
    return container


def test_get_urls_from_container_maps_ports_to_urls():
    container = _container_with_ports(
        {
            "8080/tcp": [
                {"HostIp": "0.0.0.0", "HostPort": "19051"},
                {"HostIp": "::", "HostPort": "19051"},
            ],
            "9090/tcp": None,
        }
    )

    assert docker_util.get_urls_from_container(container) == {
        8080: ["http://0.0.0.0:19051", "http://:::19051"]
    }


def test_get_urls_from_container_inspects_container_given_by_id(client):
    client.container.inspect.return_value = _container_with_ports(
        {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}]}
    )

    assert docker_util.get_urls_from_container("container_id") == {
        8080: ["http://0.0.0.0:8080"]
    }
    client.container.inspect.assert_called_once_with("container_id")


@pytest.mark.parametrize("ports", [None, "no_network_settings"])
def test_get_urls_from_container_without_ports(ports):
    container = mock.MagicMock()
    if ports == "no_network_settings":
        container.network_settings = None
    else:
        container.network_settings.ports = None

    assert docker_util.get_urls_from_container(container) == {}


def test_kill_containers_kills_matching_containers(client, caplog):
    container = mock.MagicMock()
    container.id = "abc123"
    container.config.labels = {TRUSS_DIR: "/some/truss"}
    client.container.list.return_value = [container]

    with caplog.at_level("INFO"):
        docker_util.kill_containers({TRUSS: True})

    assert "Killing Container: abc123 for /some/truss" in caplog.text
    client.container.kill.assert_called_once_with([container])


def test_kill_containers_ignores_already_stopped_containers(client):
    container = mock.MagicMock()
    container.config.labels = {"some_other_label": "value"}
    client.container.list.return_value = [container]
    client.container.kill.side_effect = DockerException(["docker", "kill"], 1)

    docker_util.kill_containers({TRUSS: True})

    client.container.kill.assert_called_once_with([container])


def test_kill_all_kills_all_truss_containers(client):
    client.container.list.return_value = []

    docker_util.kill_all()

    client.container.list.assert_called_once_with(
        filters={"label=truss": True}, all=False
    )
    client.container.kill.assert_called_once_with([])


def test_get_container_logs_forwards_arguments(client):
    logs = docker_util.get_container_logs("container_id", follow=True, stream=False)

    assert logs is client.container.logs.return_value
    client.container.logs.assert_called_once_with(
        "container_id", follow=True, stream=False
    )


def test_inspect_container(client):
    assert (
        docker_util.inspect_container("container_id")
        is client.container.inspect.return_value
    )
    client.container.inspect.assert_called_once_with("container_id")


@pytest.mark.parametrize("state", list(docker_util.DockerStates))
def test_get_container_state(client, state):
    client.container.inspect.return_value.state.status = state.value

    assert docker_util.get_container_state("container_id") == state


def test_get_container_state_rejects_unknown_state(client):
    client.container.inspect.return_value.state.status = "unknown"

    with pytest.raises(ValueError):
        docker_util.get_container_state("container_id")
