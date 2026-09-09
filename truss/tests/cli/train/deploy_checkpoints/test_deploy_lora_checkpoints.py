from truss.cli.train.deploy_checkpoints.deploy_lora_checkpoints import (
    hydrate_lora_checkpoint,
)
from truss_train.definitions import DEFAULT_LORA_RANK, LoRACheckpoint


def test_hydrate_lora_checkpoint_with_explicit_rank():
    checkpoint = {"lora_adapter_config": {"r": 8}}
    result = hydrate_lora_checkpoint("job-123", "chkpt-1", checkpoint)
    assert isinstance(result, LoRACheckpoint)
    assert result.training_job_id == "job-123"
    assert result.checkpoint_name == "chkpt-1"
    assert result.lora_details.rank == 8


def test_hydrate_lora_checkpoint_uses_default_rank():
    checkpoint = {}
    result = hydrate_lora_checkpoint("job-456", "chkpt-2", checkpoint)
    assert isinstance(result, LoRACheckpoint)
    assert result.lora_details.rank == DEFAULT_LORA_RANK


def test_hydrate_lora_checkpoint_empty_config():
    checkpoint = {"lora_adapter_config": None}
    result = hydrate_lora_checkpoint("job-789", "chkpt-3", checkpoint)
    assert result.lora_details.rank == DEFAULT_LORA_RANK
