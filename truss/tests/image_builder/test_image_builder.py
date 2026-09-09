from pathlib import Path
from unittest import mock

from truss.contexts.image_builder.image_builder import ImageBuilder


class FakeImageBuilder(ImageBuilder):
    @property
    def default_tag(self):
        return "fake:latest"

    def prepare_image_build_dir(self, build_dir):
        (build_dir / "marker").write_text("ok")


def test_build_image_with_temp_dir(tmp_path):
    builder = FakeImageBuilder()
    mock_image = mock.MagicMock()
    client = mock.MagicMock()
    client.build.return_value = mock_image

    with mock.patch(
        "truss.contexts.image_builder.image_builder.Docker.client", return_value=client
    ):
        build_dir = Path(tmp_path, "build")
        build_dir.mkdir()
        result = builder.build_image(
            build_dir=build_dir, tag="custom:tag", labels={"key": "val"}, cache=False
        )

    assert result == mock_image
    client.build.assert_called_once_with(
        str(build_dir),
        labels={"key": "val"},
        tags="custom:tag",
        cache=False,
        network=None,
        load=True,
    )


def test_build_image_uses_default_tag(tmp_path):
    builder = FakeImageBuilder()
    client = mock.MagicMock()

    with mock.patch(
        "truss.contexts.image_builder.image_builder.Docker.client", return_value=client
    ):
        build_dir = Path(tmp_path, "build")
        build_dir.mkdir()
        builder.build_image(build_dir=build_dir)

    client.build.assert_called_once_with(
        str(build_dir),
        labels={},
        tags="fake:latest",
        cache=True,
        network=None,
        load=True,
    )


def test_docker_build_command():
    builder = FakeImageBuilder()
    assert (
        builder.docker_build_command("/tmp/build")
        == "docker build /tmp/build -t fake:latest"
    )
