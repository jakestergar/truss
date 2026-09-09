from unittest.mock import Mock, patch

from truss.base.constants import TRUSS, TRUSS_DIR
from truss.util.docker import (
    DockerStates,
    _create_label_filters,
    get_container_logs,
    get_container_state,
    get_containers,
    get_images,
    get_urls_from_container,
    inspect_container,
    kill_all,
    kill_containers,
)


def test_get_urls_from_container_obj():
    container = Mock()
    container.network_settings.ports = {
        "8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}],
        "9090/tcp": None,
    }
    result = get_urls_from_container(container)
    assert result == {8080: ["http://0.0.0.0:8080"]}


def test_get_urls_from_container_no_ports():
    container = Mock()
    container.network_settings.ports = None
    assert get_urls_from_container(container) == {}


def test_get_urls_from_container_no_network_settings():
    container = Mock()
    container.network_settings = None
    assert get_urls_from_container(container) == {}


def test_get_urls_from_container_by_name():
    client = Mock()
    container = Mock()
    container.network_settings.ports = {}
    client.container.inspect.return_value = container
    with patch("truss.util.docker.Docker.client", return_value=client):
        assert get_urls_from_container("my-container") == {}
    client.container.inspect.assert_called_once_with("my-container")


def test_get_containers():
    client = Mock()
    containers = [Mock()]
    client.container.list.return_value = containers
    labels = {"foo": "bar"}
    with patch("truss.util.docker.Docker.client", return_value=client):
        result = get_containers(labels)
    assert result == containers
    client.container.list.assert_called_once_with(
        filters={"label=foo": "bar"}, all=False
    )


def test_get_images():
    client = Mock()
    images = [Mock()]
    client.image.list.return_value = images
    labels = {"foo": "bar"}
    with patch("truss.util.docker.Docker.client", return_value=client):
        result = get_images(labels)
    assert result == images
    client.image.list.assert_called_once_with(filters={"label=foo": "bar"})


def test_get_container_logs():
    client = Mock()
    client.container.logs.return_value = ["log"]
    with patch("truss.util.docker.Docker.client", return_value=client):
        assert get_container_logs("c", follow=True, stream=False) == ["log"]
    client.container.logs.assert_called_once_with("c", follow=True, stream=False)


def test_inspect_container():
    client = Mock()
    client.container.inspect.return_value = Mock()
    with patch("truss.util.docker.Docker.client", return_value=client):
        result = inspect_container("c")
    assert result is client.container.inspect.return_value
    client.container.inspect.assert_called_once_with("c")


def test_get_container_state():
    container = Mock()
    container.state.status = "running"
    with patch("truss.util.docker.inspect_container", return_value=container):
        assert get_container_state("c") == DockerStates.RUNNING


def test_create_label_filters():
    assert _create_label_filters({"a": "1", "b": "2"}) == {
        "label=a": "1",
        "label=b": "2",
    }


def test_kill_containers():
    client = Mock()
    client.container.kill.return_value = None
    container = Mock()
    container.config.labels = {TRUSS_DIR: "/tmp/truss"}
    client.container.list.return_value = [container]
    with patch("truss.util.docker.Docker.client", return_value=client):
        kill_containers({TRUSS: True})
    client.container.kill.assert_called_once_with([container])


def test_kill_all():
    client = Mock()
    client.container.list.return_value = []
    with patch("truss.util.docker.Docker.client", return_value=client):
        kill_all()
    client.container.list.assert_called_once_with(
        filters={f"label={TRUSS}": True}, all=False
    )
