import pytest

from truss_train import BasetenCheckpoint, definitions, restore_from_checkpoint


def test_example_defines_a_loadable_project():
    project = restore_from_checkpoint.project
    assert isinstance(project, definitions.TrainingProject)
    assert project.name == "new-project"
    assert project.job is restore_from_checkpoint.job


def test_example_load_checkpoint_config():
    config = restore_from_checkpoint.load_checkpoint_config
    assert config.enabled is True
    assert config.download_folder == "/tmp/custom_location"
    assert config.checkpoints == [
        restore_from_checkpoint.load_from_most_recent_checkpoint,
        restore_from_checkpoint.load_from_named_checkpoint,
        restore_from_checkpoint.load_from_loops_checkpoint,
    ]


def test_example_runtime_wires_checkpoint_configs():
    runtime = restore_from_checkpoint.job.runtime
    assert runtime.checkpointing_config is restore_from_checkpoint.checkpointing_config
    assert (
        runtime.load_checkpoint_config is restore_from_checkpoint.load_checkpoint_config
    )


def test_example_checkpoint_variants():
    latest = restore_from_checkpoint.load_most_recent_checkpoint
    assert (latest.project_name, latest.job_id) == ("first-project", "lqz4pw4")

    named = restore_from_checkpoint.load_from_named_checkpoint
    assert (named.checkpoint_name, named.job_id) == ("checkpoint-24", "lqz4pw4")

    loops = restore_from_checkpoint.load_from_loops_checkpoint
    assert (loops.run_id, loops.checkpoint_name, loops.target) == (
        "abc123",
        "step-100",
        "trainer",
    )


def test_from_latest_checkpoint_requires_project_name_or_job_id():
    with pytest.raises(ValueError, match="job_id or project_name is required"):
        BasetenCheckpoint.from_latest_checkpoint()
