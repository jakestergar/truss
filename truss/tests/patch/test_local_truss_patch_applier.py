import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from truss.templates.control.control.helpers.custom_types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.templates.control.control.helpers.errors import UnsupportedPatch
from truss.truss_handle.patch.local_truss_patch_applier import LocalTrussPatchApplier


@pytest.fixture
def applier(tmp_path):
    return LocalTrussPatchApplier(
        tmp_path, "/venv/bin/python", logging.getLogger(__name__)
    )


def test_model_code_patch_uses_configured_model_directory(applier):
    config = Mock(model_module_dir="model")
    patch_body = ModelCodePatch(action=Action.UPDATE, path="model.py", content="new")
    patch_request = Patch(PatchType.MODEL_CODE, patch_body)

    with (
        patch(
            "truss.truss_handle.patch.local_truss_patch_applier.TrussConfig.from_yaml",
            return_value=config,
        ),
        patch(
            "truss.truss_handle.patch.local_truss_patch_applier.apply_code_patch"
        ) as apply_code_patch,
    ):
        applier([patch_request])

    apply_code_patch.assert_called_once_with(
        applier._truss_dir / "model", patch_body, applier._logger
    )


@pytest.mark.parametrize(
    ("action", "command"),
    [
        (
            Action.REMOVE,
            ["/venv/bin/python", "-m", "pip", "uninstall", "-y", "requests"],
        ),
        (Action.ADD, ["/venv/bin/python", "-m", "pip", "install", "requests>=2"]),
        (Action.UPDATE, ["/venv/bin/python", "-m", "pip", "install", "requests>=2"]),
    ],
)
def test_python_requirement_patch_runs_pip(applier, action, command):
    body = PythonRequirementPatch(
        action=action,
        requirement="requests" if action is Action.REMOVE else "requests>=2",
    )
    with patch(
        "truss.truss_handle.patch.local_truss_patch_applier.subprocess.run"
    ) as run:
        applier([Patch(PatchType.PYTHON_REQUIREMENT, body)])

    run.assert_called_once_with(command, check=True)


def test_system_package_patch_logs_unsupported_action(applier, caplog):
    body = SystemPackagePatch(action=Action.ADD, package="curl")
    with caplog.at_level(logging.INFO):
        applier([Patch(PatchType.SYSTEM_PACKAGE, body)])
    assert "not supported for local server" in caplog.text


def test_unknown_patch_body_raises_unsupported_patch(applier):
    request = Mock(type=PatchType.CONFIG, body=Mock())
    with pytest.raises(UnsupportedPatch, match="Unknown patch type"):
        applier([request])


def test_unknown_requirement_action_raises_value_error(applier):
    body = PythonRequirementPatch(
        action=SimpleNamespace(value="INVALID"), requirement="pkg"
    )
    with pytest.raises(ValueError, match="Unknown python requirement patch action"):
        applier._apply_python_requirement_patch(body)
