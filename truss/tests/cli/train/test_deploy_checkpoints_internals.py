import json
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import rich_click as click

from truss.cli.train.deploy_checkpoints.deploy_checkpoints import (
    _build_inference_template_request,
    _ensure_checkpoint_details,
    _ensure_compute_spec,
    _ensure_runtime_config,
    _get_accelerator_if_specified,
    _get_base_model_id,
    _get_checkpoint_ids_to_deploy,
    _get_instance_type_id,
    _get_model_name,
    _get_truss_config_from_result,
    _guard_interactive,
    _hydrate_checkpoints,
    _hydrate_deploy_config,
    _model_name_from_checkpoint_model_ref,
    _process_user_provided_checkpoints,
    _select_multiple_checkpoints,
    _validate_base_model_id,
    _validate_selected_checkpoints,
    create_model_version_from_inference_template,
    get_hf_secret_name,
    hydrate_checkpoint,
)
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss.remote.baseten.api import InstanceTypeV1
from truss_train.definitions import (
    CheckpointList,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
    FullCheckpoint,
    LoRACheckpoint,
    ModelWeightsFormat,
    SecretReference,
    WhisperCheckpoint,
)


@pytest.fixture(autouse=True)
def interactive():
    with patch(
        "truss.cli.utils.common.check_is_interactive", return_value=True
    ) as mock:
        yield mock


@pytest.fixture
def remote():
    value = MagicMock()
    value.remote_url = "https://app.baseten.co"
    return value


def prompt(value):
    result = MagicMock()
    result.execute.return_value = value
    return result


def instance(
    identifier="instance",
    *,
    gpu_count=0,
    node_count=1,
    gpu_type=None,
    cpu=1000,
    memory=2048,
):
    return InstanceTypeV1(
        id=identifier,
        name=identifier,
        displayName=identifier,
        gpuCount=gpu_count,
        default=False,
        gpuMemory=None,
        nodeCount=node_count,
        gpuType=gpu_type,
        millicpuLimit=cpu,
        memoryLimit=memory,
        price=None,
        limitedCapacity=None,
    )


def checkpoint(checkpoint_type="lora", checkpoint_id="cp"):
    if checkpoint_type == "full":
        return FullCheckpoint(training_job_id="job", checkpoint_name=checkpoint_id)
    if checkpoint_type == "whisper":
        return WhisperCheckpoint(training_job_id="job", checkpoint_name=checkpoint_id)
    return LoRACheckpoint(training_job_id="job", checkpoint_name=checkpoint_id)


def test_build_request_and_result_parsing(remote):
    # CheckpointList validation intentionally rejects mixing both checkpoint sources.
    config = DeployCheckpointsConfigComplete.model_construct(
        checkpoint_details=CheckpointList.model_construct(
            download_folder="/tmp/training_checkpoints",
            base_model_id=None,
            checkpoints=[checkpoint()],
            loops_checkpoint_ids=["loops-pk"],
        ),
        model_name="deployment",
        runtime=DeployCheckpointsRuntime(
            environment_variables={"HF_TOKEN": SecretReference(name="secret"), "N": "3"}
        ),
        compute=Compute(cpu_count=1, memory="1Gi"),
    )
    remote.api.get_instance_types.return_value = [
        instance("cpu", cpu=1000, memory=2**30)
    ]
    request = _build_inference_template_request(config, remote, dry_run=True)
    assert request["metadata"] == {"oracle_name": "deployment"}
    assert request["dry_run"] is True
    assert request["instance_type_id"] == "cpu"
    assert request["weights_sources"][0]["weight_source_type"] == "B10_CHECKPOINTING"
    assert (
        request["weights_sources"][1]["weight_source_type"] == "B10_LOOPS_CHECKPOINTING"
    )
    assert request["inference_stack"]["environment_variables"] == [
        {"name": "HF_TOKEN", "value": "secret", "is_secret_reference": True},
        {"name": "N", "value": "3", "is_secret_reference": False},
    ]


def test_create_model_version_success_and_dry_run(remote):
    config = DeployCheckpointsConfigComplete(
        checkpoint_details=CheckpointList(checkpoints=[checkpoint()]),
        model_name="deployment",
        runtime=DeployCheckpointsRuntime(),
        compute=Compute(cpu_count=1, memory="1Gi"),
    )
    remote.api.get_instance_types.return_value = [instance("cpu", memory=2**30)]
    remote.api.create_model_version_from_inference_template.return_value = {
        "model_version": {"name": "deployment", "id": "v1", "model_id": "m1"},
        "truss_config": json.dumps({"model_name": "deployment"}),
    }
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints._hydrate_deploy_config",
        return_value=config,
    ):
        result = create_model_version_from_inference_template(
            remote, config, None, None, None, False
        )
    assert result.model_version.id == "v1"
    assert result.truss_config.model_name == "deployment"
    remote.api.create_model_version_from_inference_template.assert_called_once()

    remote.api.create_model_version_from_inference_template.return_value = {}
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints._hydrate_deploy_config",
            return_value=config,
        ),
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.console.print"
        ) as output,
    ):
        result = create_model_version_from_inference_template(
            remote, config, None, None, None, True
        )
    assert result.model_version is None
    assert not any("Unexpected response" in str(call) for call in output.call_args_list)


def test_create_model_version_warning_and_error(remote):
    config = DeployCheckpointsConfigComplete(
        checkpoint_details=CheckpointList(checkpoints=[checkpoint()]),
        model_name="deployment",
        runtime=DeployCheckpointsRuntime(),
        compute=Compute(cpu_count=1, memory="1Gi"),
    )
    remote.api.get_instance_types.return_value = [instance("cpu", memory=2**30)]
    remote.api.create_model_version_from_inference_template.return_value = {}
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints._hydrate_deploy_config",
            return_value=config,
        ),
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.console.print"
        ) as output,
    ):
        create_model_version_from_inference_template(
            remote, config, None, None, None, False
        )
    assert any("Unexpected response" in str(call) for call in output.call_args_list)
    remote.api.create_model_version_from_inference_template.side_effect = RuntimeError(
        "boom"
    )
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints._hydrate_deploy_config",
            return_value=config,
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        create_model_version_from_inference_template(
            remote, config, None, None, None, False
        )


def test_get_truss_config_branches():
    result = _get_truss_config_from_result(
        {"truss_config": json.dumps({"model_name": "x"})}
    )
    assert result.model_name == "x"
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.console.print"
    ) as output:
        assert _get_truss_config_from_result({}) is None
    output.assert_called_once()


@pytest.mark.parametrize(
    ("compute", "types", "expected"),
    [
        (
            Compute(cpu_count=1, memory="1Gi"),
            [
                instance("large", cpu=2000, memory=2**30),
                instance("small", cpu=1000, memory=2**30),
            ],
            "small",
        ),
        (
            Compute(
                cpu_count=1,
                memory="1Gi",
                accelerator={"accelerator": "H100", "count": 2},
            ),
            [
                instance("two", gpu_count=2, gpu_type="H100"),
                instance("four", gpu_count=4, gpu_type="H100"),
                instance("other", gpu_count=1, gpu_type="A100"),
            ],
            "two",
        ),
    ],
)
def test_get_instance_type_id(compute, types, expected, remote):
    remote.api.get_instance_types.return_value = types + [
        instance("multi", node_count=2)
    ]
    assert _get_instance_type_id(compute, remote) == expected


def test_get_instance_type_id_errors(remote):
    remote.api.get_instance_types.return_value = [
        instance("gpu", gpu_count=1, gpu_type="A100")
    ]
    with pytest.raises(ValueError, match="Unable to find single-node"):
        _get_instance_type_id(Compute(cpu_count=99, memory="1Gi"), remote)
    with pytest.raises(ValueError, match="Unable to find single-node"):
        _get_instance_type_id(
            Compute(accelerator={"accelerator": "H100", "count": 1}), remote
        )


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
        ("openai/whisper-large-v3", "whisper-large-v3"),
        ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
    ],
)
def test_model_name_from_checkpoint_model_ref(ref, expected):
    assert _model_name_from_checkpoint_model_ref(ref) == expected


def test_validate_and_get_model_name():
    _validate_base_model_id("org/model", ModelWeightsFormat.LORA)
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text",
        return_value=prompt("model"),
    ):
        assert _get_model_name(ModelWeightsFormat.LORA, "org/model") == "model"
    with pytest.raises(ValueError, match="Unable to infer base model"):
        _validate_base_model_id(None, ModelWeightsFormat.LORA)


def test_get_model_name_non_lora_and_default():
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text",
        return_value=prompt("named"),
    ) as text:
        assert _get_model_name(ModelWeightsFormat.FULL, None) == "named"
    assert text.call_args.kwargs["default"] == ""


def test_hydrate_deploy_config_training_and_loops(remote):
    remote.api.get_instance_types.return_value = [instance("cpu", memory=4096)]
    remote.api.search_training_jobs.return_value = [
        {"id": "job", "training_project": {"id": "project"}}
    ]
    remote.api.list_training_job_checkpoints.return_value = {
        "checkpoints": [
            {"checkpoint_id": "a", "checkpoint_type": "lora", "base_model": "org/model"}
        ]
    }
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox",
            return_value=prompt(["a"]),
        ),
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text",
            return_value=prompt("name"),
        ),
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select",
            return_value=prompt(None),
        ),
    ):
        result = _hydrate_deploy_config(
            DeployCheckpointsConfig(), remote, None, "job", None
        )
    assert result.model_name == "name"
    assert result.checkpoint_details.checkpoints[0].checkpoint_name == "a"

    remote.api.list_loops_runs.return_value = [
        {"id": "run", "base_model": "org/loop-model"}
    ]
    remote.api.list_loops_checkpoints.return_value = {
        "checkpoints": [{"id": "pk", "checkpoint_id": "sampler", "target": "sampler"}]
    }
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text",
            return_value=prompt("loop-model"),
        ),
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select",
            return_value=prompt(None),
        ),
    ):
        result = _hydrate_deploy_config(
            DeployCheckpointsConfig(), remote, None, None, "run", is_loops_command=True
        )
    assert result.checkpoint_details.loops_checkpoint_ids == ["pk"]
    assert result.model_name == "loop-model"

    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text"
    ) as text:
        result = _hydrate_deploy_config(
            DeployCheckpointsConfig(
                model_name="configured-loops",
                checkpoint_details=CheckpointList(loops_checkpoint_ids=["pk"]),
                runtime=DeployCheckpointsRuntime(environment_variables={"X": "y"}),
            ),
            remote,
            None,
            None,
            None,
            is_loops_command=True,
        )
    assert result.model_name == "configured-loops"
    text.assert_not_called()

    remote.api.search_training_jobs.return_value = [
        {"id": "job", "training_project": {"id": "project"}}
    ]
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text"
    ) as text:
        result = _hydrate_deploy_config(
            DeployCheckpointsConfig(
                model_name="configured-training",
                checkpoint_details=CheckpointList(
                    checkpoints=[
                        LoRACheckpoint(training_job_id="job", checkpoint_name="a")
                    ]
                ),
                runtime=DeployCheckpointsRuntime(environment_variables={"X": "y"}),
            ),
            remote,
            None,
            None,
            None,
        )
    assert result.model_name == "configured-training"
    text.assert_not_called()


def test_hydrate_deploy_config_rejects_flags_and_mismatches(remote):
    config = DeployCheckpointsConfig(
        checkpoint_details=CheckpointList(loops_checkpoint_ids=["pk"])
    )
    with pytest.raises(click.UsageError, match="project-id"):
        _hydrate_deploy_config(
            config, remote, "project", None, None, is_loops_command=True
        )
    with pytest.raises(click.UsageError, match="does not accept Loops"):
        _hydrate_deploy_config(config, remote, None, None, None)
    with pytest.raises(click.UsageError, match="does not accept training-job"):
        _hydrate_deploy_config(
            DeployCheckpointsConfig(
                checkpoint_details=CheckpointList(checkpoints=[checkpoint()])
            ),
            remote,
            None,
            None,
            None,
            is_loops_command=True,
        )


def test_process_user_checkpoints_deduplicates_and_not_found(remote):
    details = CheckpointList(
        checkpoints=[
            LoRACheckpoint(training_job_id="job", checkpoint_name="a"),
            LoRACheckpoint(training_job_id="job", checkpoint_name="b"),
        ]
    )
    remote.api.search_training_jobs.return_value = [{"training_project": {"id": "p"}}]
    remote.api.list_training_job_checkpoints.return_value = {"checkpoints": []}
    assert _process_user_provided_checkpoints(details, remote) is details
    remote.api.search_training_jobs.assert_called_once_with(job_id="job")
    remote.api.search_training_jobs.reset_mock()
    remote.api.search_training_jobs.return_value = []
    with pytest.raises(click.UsageError, match="Training job missing not found"):
        _process_user_provided_checkpoints(
            CheckpointList(
                checkpoints=[
                    LoRACheckpoint(training_job_id="missing", checkpoint_name="a")
                ]
            ),
            remote,
        )


def test_checkpoint_selection_and_validation():
    with pytest.raises(click.UsageError, match="No checkpoints"):
        _get_checkpoint_ids_to_deploy([], OrderedDict())
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox",
            return_value=prompt([]),
        ),
        pytest.raises(click.UsageError, match="At least one"),
    ):
        _select_multiple_checkpoints(["a", "b"])
    with pytest.raises(ValueError, match="Unable to infer"):
        _validate_selected_checkpoints(["a"], OrderedDict([("a", {})]))
    for kind, message in [
        ("full", "Full checkpoints"),
        ("whisper", "Whisper checkpoints"),
    ]:
        with pytest.raises(ValueError, match=message):
            _validate_selected_checkpoints(
                ["a", "b"],
                OrderedDict(
                    [
                        ("a", {"checkpoint_type": kind}),
                        ("b", {"checkpoint_type": "lora"}),
                    ]
                ),
            )
    _validate_selected_checkpoints(
        ["a", "b"],
        OrderedDict(
            [("a", {"checkpoint_type": "lora"}), ("b", {"checkpoint_type": "lora"})]
        ),
    )


def test_hydrate_checkpoints_latest_and_unsupported():
    response = OrderedDict(
        [
            ("first", {"checkpoint_type": "lora", "base_model": "org/model"}),
            ("last", {"checkpoint_type": "full", "base_model": "org/model"}),
        ]
    )
    result = _hydrate_checkpoints("job", "latest", response)
    assert result.checkpoint_name == "last"
    assert isinstance(
        hydrate_checkpoint("job", "whisper", {"checkpoint_type": "whisper"}, "whisper"),
        WhisperCheckpoint,
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint type"):
        hydrate_checkpoint("job", "x", {"checkpoint_type": "other"}, "other")


def test_compute_and_accelerator_paths(remote):
    supplied = Compute(accelerator={"accelerator": "A100", "count": 1})
    assert _get_accelerator_if_specified(supplied, remote) is supplied
    remote.api.get_instance_types.return_value = [instance("cpu")]
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.console.print"
    ) as output:
        result = _ensure_compute_spec(None, remote)
    assert result.accelerator is None
    assert "No GPU instance types available" in str(output.call_args)

    remote.api.get_instance_types.return_value = [
        instance("gpu", gpu_count=2, gpu_type="A100")
    ]
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select",
        side_effect=[prompt("A100"), prompt(2)],
    ):
        result = _ensure_compute_spec(None, remote)
    assert result.accelerator.accelerator.value == "A100"
    assert result.accelerator.count == 2


@pytest.mark.parametrize(
    ("user_input", "data", "expected"),
    [
        ("provided", {}, "provided"),
        (None, {"base_model": "org/model"}, "org/model"),
        (None, {"checkpoint_type": "full"}, None),
        (None, {"checkpoint_type": "whisper"}, None),
        (None, {"checkpoint_type": "lora"}, "prompted"),
    ],
)
def test_get_base_model_id(user_input, data, expected):
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text",
        return_value=prompt("prompted"),
    ):
        assert _get_base_model_id(user_input, data) == expected


def test_get_base_model_id_empty_prompt():
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text",
            return_value=prompt(""),
        ),
        pytest.raises(click.UsageError, match="Base model id is required"),
    ):
        _get_base_model_id(None, {"checkpoint_type": "lora"})


def test_runtime_config_and_hf_secret():
    assert _ensure_runtime_config(
        DeployCheckpointsRuntime(environment_variables={"X": "y"})
    ).environment_variables == {"X": "y"}
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select",
        return_value=prompt("hf"),
    ):
        result = _ensure_runtime_config(None)
    assert result.environment_variables["HF_TOKEN"].name == "hf"
    assert get_hf_secret_name(SecretReference(name="secret")) == "secret"
    assert get_hf_secret_name("plain") == "plain"
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select",
            return_value=prompt("custom"),
        ),
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text",
            return_value=prompt("custom-secret"),
        ),
    ):
        assert get_hf_secret_name(None) == "custom-secret"
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select",
            return_value=prompt(None),
        ),
        patch("truss.cli.train.deploy_checkpoints.deploy_checkpoints.console.print"),
    ):
        assert get_hf_secret_name(None) is None


def test_guard_interactive():
    with (
        patch("truss.cli.utils.common.check_is_interactive", return_value=False),
        pytest.raises(click.UsageError, match="non-interactive"),
    ):
        _guard_interactive("value")


def test_ensure_checkpoint_details_prompt(remote):
    remote.api.search_training_jobs.return_value = [
        {"id": "job", "training_project": {"id": "project"}}
    ]
    remote.api.list_training_job_checkpoints.return_value = {
        "checkpoints": [
            {"checkpoint_id": "a", "checkpoint_type": "full", "base_model": "org/model"}
        ]
    }
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox",
        return_value=prompt(["a"]),
    ):
        result = _ensure_checkpoint_details(remote, None, None, "job")
    assert result.checkpoints[0].model_weight_format == ModelWeightsFormat.FULL


def test_ensure_checkpoint_details_processes_configured_checkpoints(remote):
    details = CheckpointList(
        checkpoints=[LoRACheckpoint(training_job_id="job", checkpoint_name="a")]
    )
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints._process_user_provided_checkpoints",
        return_value=details,
    ) as process:
        assert _ensure_checkpoint_details(remote, details, None, None) is details
    process.assert_called_once_with(details, remote)
