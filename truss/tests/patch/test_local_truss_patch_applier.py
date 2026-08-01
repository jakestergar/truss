import logging
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from truss.templates.control.control.helpers.custom_types import (
    Action,
    ModelCodePatch,
    PackagePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.templates.control.control.helpers.errors import UnsupportedPatch
from truss.truss_handle.patch.local_truss_patch_applier import LocalTrussPatchApplier

TEST_LOGGER = logging.getLogger("test_logger")


def test_model_code_patch(custom_model_truss_dir: Path):
    applier = LocalTrussPatchApplier(custom_model_truss_dir, "env", TEST_LOGGER)

    applier(
        [
            Patch(
                type=PatchType.MODEL_CODE,
                body=ModelCodePatch(
                    action=Action.UPDATE, path="nested/model.py", content="updated"
                ),
            )
        ]
    )

    assert (
        custom_model_truss_dir / "model" / "nested" / "model.py"
    ).read_text() == "updated"


@pytest.mark.parametrize(
    ("action", "pip_action", "extra_args"),
    [
        (Action.ADD, "install", []),
        (Action.UPDATE, "install", []),
        (Action.REMOVE, "uninstall", ["-y"]),
    ],
)
def test_python_requirement_patch(
    custom_model_truss_dir: Path, action: Action, pip_action: str, extra_args: list[str]
):
    requirement = "numpy==1.26.4"
    applier = LocalTrussPatchApplier(
        custom_model_truss_dir, "/venv/bin/python", TEST_LOGGER
    )
    patch = PythonRequirementPatch(action=action, requirement=requirement)

    with mock.patch(
        "truss.truss_handle.patch.local_truss_patch_applier.subprocess"
    ) as subprocess_mock:
        applier([Patch(type=PatchType.PYTHON_REQUIREMENT, body=patch)])

    subprocess_mock.run.assert_called_once_with(
        ["/venv/bin/python", "-m", "pip", pip_action, *extra_args, requirement],
        check=True,
    )


def test_python_requirement_patch_invalid_action(custom_model_truss_dir: Path):
    applier = LocalTrussPatchApplier(custom_model_truss_dir, "env", TEST_LOGGER)
    patch = PythonRequirementPatch(action=Action.ADD, requirement="numpy")
    patch.action = SimpleNamespace(value="INVALID")

    with pytest.raises(ValueError, match="Unknown python requirement patch action"):
        applier._apply_python_requirement_patch(patch)


def test_system_package_patch_logs_and_does_nothing(
    custom_model_truss_dir: Path, caplog: pytest.LogCaptureFixture
):
    applier = LocalTrussPatchApplier(custom_model_truss_dir, "env", TEST_LOGGER)

    with caplog.at_level(logging.INFO, logger=TEST_LOGGER.name):
        applier(
            [
                Patch(
                    type=PatchType.SYSTEM_PACKAGE,
                    body=SystemPackagePatch(action=Action.ADD, package="curl"),
                )
            ]
        )

    assert "System package patches are not supported for local server" in caplog.text


def test_unsupported_patch_raises(custom_model_truss_dir: Path):
    applier = LocalTrussPatchApplier(custom_model_truss_dir, "env", TEST_LOGGER)
    patch = Patch(
        type=PatchType.PACKAGE,
        body=PackagePatch(action=Action.ADD, path="package.py", content="pass"),
    )

    with pytest.raises(UnsupportedPatch, match="Unknown patch type PatchType.PACKAGE"):
        applier([patch])


def test_empty_patch_list_is_no_op(custom_model_truss_dir: Path):
    applier = LocalTrussPatchApplier(custom_model_truss_dir, "env", TEST_LOGGER)

    with mock.patch(
        "truss.truss_handle.patch.local_truss_patch_applier.subprocess"
    ) as subprocess_mock:
        applier([])

    subprocess_mock.run.assert_not_called()
