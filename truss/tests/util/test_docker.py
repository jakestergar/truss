from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from python_on_whales.exceptions import DockerException

from truss.base.constants import TRUSS, TRUSS_DIR
from truss.util import docker as docker_util


@pytest.fixture(autouse=True)
def reset_docker_client():
    original_client = docker_util.Docker._client
    docker_util.Docker._client = None
    yield
    docker_util.Docker._client = original_client


def test_client_uses_sudo_docker_client():
    client = Mock()
    with (
        patch("truss.util.docker.LocalConfigHandler.get_config") as get_config,
        patch("python_on_whales.DockerClient", return_value=client) as docker_client,
        patch("python_on_whales.docker") as docker,
    ):
        get_config.return_value.use_sudo = True

        assert docker_util.Docker.client() is client
        assert docker_util.Docker.client() is client

    docker_client.assert_called_once_with(client_call=["sudo", "docker"])
    docker.assert_not_called()


def test_client_uses_docker_module_without_sudo():
    docker = Mock()
    with (
        patch("truss.util.docker.LocalConfigHandler.get_config") as get_config,
        patch("python_on_whales.docker", docker),
        patch("python_on_whales.DockerClient") as docker_client,
    ):
        get_config.return_value.use_sudo = False

        assert docker_util.Docker.client() is docker

    docker_client.assert_not_called()


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ({}, {}),
        (
            {TRUSS: True, TRUSS_DIR: "/tmp/truss"},
            {"label=truss": True, "label=truss_dir": "/tmp/truss"},
        ),
    ],
)
def test_create_label_filters(labels, expected):
    assert docker_util._create_label_filters(labels) == expected


def test_get_containers_passes_label_filters_and_all():
    client = Mock()
    client.container.list.return_value = ["container"]
    with patch.object(docker_util.Docker, "client", return_value=client):
        assert docker_util.get_containers({"app": "model"}, all=True) == ["container"]

    client.container.list.assert_called_once_with(
        filters={"label=app": "model"}, all=True
    )


def test_get_images_passes_label_filters():
    client = Mock()
    client.image.list.return_value = ["image"]
    with patch.object(docker_util.Docker, "client", return_value=client):
        assert docker_util.get_images({"app": "model"}) == ["image"]

    client.image.list.assert_called_once_with(filters={"label=app": "model"})


@pytest.mark.parametrize(
    ("container", "expected"),
    [
        (SimpleNamespace(network_settings=None), {}),
        (SimpleNamespace(network_settings=SimpleNamespace(ports=None)), {}),
        (
            SimpleNamespace(
                network_settings=SimpleNamespace(
                    ports={
                        "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18080"}],
                        "9000/udp": None,
                        "9001/tcp": [
                            {"HostIp": "0.0.0.0", "HostPort": "19001"},
                            {"HostIp": "::", "HostPort": "19001"},
                        ],
                    }
                )
            ),
            {
                8080: ["http://127.0.0.1:18080"],
                9001: ["http://0.0.0.0:19001", "http://:::19001"],
            },
        ),
    ],
)
def test_get_urls_from_container_object(container, expected):
    assert docker_util.get_urls_from_container(container) == expected


def test_get_urls_from_container_inspects_string():
    client = Mock()
    inspected = SimpleNamespace(
        network_settings=SimpleNamespace(
            ports={"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "18080"}]}
        )
    )
    client.container.inspect.return_value = inspected
    with patch.object(docker_util.Docker, "client", return_value=client):
        assert docker_util.get_urls_from_container("container-id") == {
            8080: ["http://0.0.0.0:18080"]
        }

    client.container.inspect.assert_called_once_with("container-id")


def test_kill_containers_logs_labeled_container_and_kills():
    client = Mock()
    container = SimpleNamespace(
        id="container-id", config=SimpleNamespace(labels={TRUSS_DIR: "/tmp/truss"})
    )
    with (
        patch.object(docker_util.Docker, "client", return_value=client),
        patch.object(
            docker_util, "get_containers", return_value=[container]
        ) as get_containers,
        patch("truss.util.docker.logging.info") as log,
    ):
        docker_util.kill_containers({TRUSS: True})

    get_containers.assert_called_once_with({TRUSS: True})
    client.container.kill.assert_called_once_with([container])
    log.assert_called_once_with("Killing Container: container-id for /tmp/truss")


def test_kill_containers_swallows_docker_exception():
    client = Mock()
    container = SimpleNamespace(
        id="id", config=SimpleNamespace(labels={TRUSS_DIR: "dir"})
    )
    client.container.kill.side_effect = DockerException(["docker", "kill"], 1)
    with (
        patch.object(docker_util.Docker, "client", return_value=client),
        patch.object(docker_util, "get_containers", return_value=[container]),
    ):
        docker_util.kill_containers({TRUSS: True})


def test_kill_containers_handles_unlabeled_container():
    client = Mock()
    container = SimpleNamespace(
        id="id", config=SimpleNamespace(labels={"other": "value"})
    )
    with (
        patch.object(docker_util.Docker, "client", return_value=client),
        patch.object(docker_util, "get_containers", return_value=[container]),
    ):
        docker_util.kill_containers({})

    client.container.kill.assert_called_once_with([container])


def test_get_container_logs_passthrough():
    client = Mock()
    client.container.logs.return_value = iter(["log"])
    with patch.object(docker_util.Docker, "client", return_value=client):
        assert list(
            docker_util.get_container_logs("id", follow=True, stream=False)
        ) == ["log"]

    client.container.logs.assert_called_once_with("id", follow=True, stream=False)


def test_docker_states_values():
    assert {state.value for state in docker_util.DockerStates} == {
        "created",
        "running",
        "paused",
        "restarting",
        "oomkilled",
        "dead",
        "exited",
    }


def test_inspect_container():
    client = Mock()
    inspected = object()
    client.container.inspect.return_value = inspected
    with patch.object(docker_util.Docker, "client", return_value=client):
        assert docker_util.inspect_container("id") is inspected

    client.container.inspect.assert_called_once_with("id")


@pytest.mark.parametrize("status", list(docker_util.DockerStates))
def test_get_container_state(status):
    client = Mock()
    client.container.inspect.return_value = SimpleNamespace(
        state=SimpleNamespace(status=status.value)
    )
    with patch.object(docker_util.Docker, "client", return_value=client):
        assert docker_util.get_container_state("id") is status


def test_kill_all():
    with patch("truss.util.docker.kill_containers") as kill_containers:
        docker_util.kill_all()

    kill_containers.assert_called_once_with({TRUSS: True})
