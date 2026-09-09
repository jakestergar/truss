from truss.truss_handle.patch.constants import PATCHABLE_STATUSES


def test_patchable_statuses():
    assert isinstance(PATCHABLE_STATUSES, list)
    assert "MODEL_READY" in PATCHABLE_STATUSES
    assert "MODEL_LOADING" in PATCHABLE_STATUSES
