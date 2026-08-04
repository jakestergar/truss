import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest

from truss.templates.control.control.helpers.custom_types import (
    Action,
    ConfigPatch,
    ModelCodePatch,
    Patch,
    PatchBody,
    PatchType,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.templates.control.control.helpers.errors import UnsupportedPatch
from truss.truss_handle.patch.local_truss_patch_applier import LocalTrussPatchApplier

TEST_LOGGER = logging.getLogger("test_logger")
ENV_EXE = "/some/env/bin/python"


class UnknownAction(Enum):
    BOGUS = "BOGUS"


@dataclass
class UnknownPatchBody(PatchBody):
    def to_dict(self):
        return {"action": self.action.value}


@pytest.fixture
def applier(custom_model_truss_dir: Path) -> LocalTrussPatchApplier:
    return LocalTrussPatchApplier(custom_model_truss_dir, ENV_EXE, TEST_LOGGER)


def test_no_patches_is_a_no_op(applier: LocalTrussPatchApplier):
    with mock_patch.object(subprocess, "run") as mock_run:
        applier([])
    mock_run.assert_not_called()


@pytest.mark.parametrize("action", [Action.ADD, Action.UPDATE])
def test_model_code_patch_writes_file(
    applier: LocalTrussPatchApplier, custom_model_truss_dir: Path, action: Action
):
    applier(
        [
            Patch(
                type=PatchType.MODEL_CODE,
                body=ModelCodePatch(
                    action=action, path="subdir/new_model.py", content="test_content"
                ),
            )
        ]
    )
    assert (
        custom_model_truss_dir / "model" / "subdir" / "new_model.py"
    ).read_text() == "test_content"


def test_model_code_patch_removes_file(
    applier: LocalTrussPatchApplier, custom_model_truss_dir: Path
):
    model_file = custom_model_truss_dir / "model" / "model.py"
    assert model_file.exists()
    applier(
        [
            Patch(
                type=PatchType.MODEL_CODE,
                body=ModelCodePatch(action=Action.REMOVE, path="model.py"),
            )
        ]
    )
    assert not model_file.exists()


def test_model_code_patch_respects_configured_model_module_dir(
    applier: LocalTrussPatchApplier, custom_model_truss_dir: Path
):
    config_path = custom_model_truss_dir / "config.yaml"
    config_path.write_text(
        config_path.read_text() + "\nmodel_module_dir: custom_model_dir\n"
    )
    applier(
        [
            Patch(
                type=PatchType.MODEL_CODE,
                body=ModelCodePatch(
                    action=Action.ADD, path="model.py", content="other_content"
                ),
            )
        ]
    )
    assert (
        custom_model_truss_dir / "custom_model_dir" / "model.py"
    ).read_text() == "other_content"


@pytest.mark.parametrize("action", [Action.ADD, Action.UPDATE])
def test_python_requirement_patch_installs(
    applier: LocalTrussPatchApplier, action: Action
):
    req = "git+https://github.com/huggingface/transformers.git"
    with mock_patch.object(subprocess, "run") as mock_run:
        applier(
            [
                Patch(
                    type=PatchType.PYTHON_REQUIREMENT,
                    body=PythonRequirementPatch(action=action, requirement=req),
                )
            ]
        )
    mock_run.assert_called_once_with([ENV_EXE, "-m", "pip", "install", req], check=True)


def test_python_requirement_patch_uninstalls(applier: LocalTrussPatchApplier):
    with mock_patch.object(subprocess, "run") as mock_run:
        applier(
            [
                Patch(
                    type=PatchType.PYTHON_REQUIREMENT,
                    body=PythonRequirementPatch(
                        action=Action.REMOVE, requirement="requests"
                    ),
                )
            ]
        )
    mock_run.assert_called_once_with(
        [ENV_EXE, "-m", "pip", "uninstall", "-y", "requests"], check=True
    )


def test_python_requirement_patch_unknown_action(applier: LocalTrussPatchApplier):
    body = PythonRequirementPatch(action=Action.ADD, requirement="requests")
    body.action = UnknownAction.BOGUS
    with mock_patch.object(subprocess, "run") as mock_run:
        with pytest.raises(ValueError, match="Unknown python requirement patch action"):
            applier([Patch(type=PatchType.PYTHON_REQUIREMENT, body=body)])
    mock_run.assert_not_called()


def test_python_requirement_patch_propagates_pip_failure(
    applier: LocalTrussPatchApplier,
):
    with mock_patch.object(
        subprocess, "run", side_effect=subprocess.CalledProcessError(1, "pip")
    ):
        with pytest.raises(subprocess.CalledProcessError):
            applier(
                [
                    Patch(
                        type=PatchType.PYTHON_REQUIREMENT,
                        body=PythonRequirementPatch(
                            action=Action.ADD, requirement="requests"
                        ),
                    )
                ]
            )


def test_system_package_patch_is_ignored(applier: LocalTrussPatchApplier):
    with mock_patch.object(subprocess, "run") as mock_run:
        applier(
            [
                Patch(
                    type=PatchType.SYSTEM_PACKAGE,
                    body=SystemPackagePatch(action=Action.ADD, package="curl"),
                )
            ]
        )
    mock_run.assert_not_called()


def test_unsupported_patch_type(applier: LocalTrussPatchApplier):
    with pytest.raises(UnsupportedPatch, match="Unknown patch type"):
        applier(
            [
                Patch(
                    type=PatchType.CONFIG,
                    body=ConfigPatch(action=Action.UPDATE, config={}),
                )
            ]
        )


def test_unknown_patch_body(applier: LocalTrussPatchApplier):
    with pytest.raises(UnsupportedPatch, match="Unknown patch type"):
        applier([Patch(type=PatchType.DATA, body=UnknownPatchBody(action=Action.ADD))])


def test_multiple_patches_are_applied_in_order(
    applier: LocalTrussPatchApplier, custom_model_truss_dir: Path
):
    with mock_patch.object(subprocess, "run") as mock_run:
        applier(
            [
                Patch(
                    type=PatchType.MODEL_CODE,
                    body=ModelCodePatch(
                        action=Action.UPDATE, path="model.py", content="new_content"
                    ),
                ),
                Patch(
                    type=PatchType.SYSTEM_PACKAGE,
                    body=SystemPackagePatch(action=Action.ADD, package="curl"),
                ),
                Patch(
                    type=PatchType.PYTHON_REQUIREMENT,
                    body=PythonRequirementPatch(
                        action=Action.REMOVE, requirement="requests"
                    ),
                ),
            ]
        )
    assert (custom_model_truss_dir / "model" / "model.py").read_text() == "new_content"
    mock_run.assert_called_once_with(
        [ENV_EXE, "-m", "pip", "uninstall", "-y", "requests"], check=True
    )
