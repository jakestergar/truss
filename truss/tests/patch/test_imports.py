def test_hash_reexports():
    from truss.patch.hash import directory_content_hash

    assert callable(directory_content_hash)


def test_signature_reexports():
    from truss.patch.signature import calc_truss_signature

    assert callable(calc_truss_signature)


def test_truss_dir_patch_applier_reexports():
    from truss.patch.truss_dir_patch_applier import TrussDirPatchApplier

    assert TrussDirPatchApplier is not None
