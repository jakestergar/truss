def test_restore_from_checkpoint_importable():
    from truss_train import restore_from_checkpoint

    assert hasattr(restore_from_checkpoint, "project")
    assert restore_from_checkpoint.project.name == "new-project"
    assert restore_from_checkpoint.job.image.base_image == (
        "ghcr.io/baseten-ai/truss-train-base:latest"
    )
