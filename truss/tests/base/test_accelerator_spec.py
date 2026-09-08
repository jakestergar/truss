"""Unit tests for GPU/accelerator config parsing in `truss.base.truss_config`.

Covers `Accelerator`, `AcceleratorSpec` (string/dict/instance parsing, serialization,
json-schema hook, assignment validation) and the GPU-adjacent parts of `Resources`.
"""

import pathlib

import pydantic
import pytest
import yaml

from truss.base.truss_config import Accelerator, AcceleratorSpec, Resources, TrussConfig


@pytest.mark.parametrize(
    "bad_spec, match",
    [
        ("", "Accelerator type cannot be empty."),
        ("   ", "Accelerator type cannot be empty."),
        (":", "Accelerator type cannot be empty."),
        (":2", "Accelerator type cannot be empty."),
    ],
)
def test_from_string_spec_empty_accelerator_type(bad_spec, match):
    with pytest.raises(pydantic.ValidationError, match=match):
        AcceleratorSpec.model_validate(bad_spec)


@pytest.mark.parametrize(
    "bad_spec", ["A100:", "A100:0", "A100:-1", "A100:2.5", "A100: 2", "A100:two"]
)
def test_from_string_spec_invalid_count(bad_spec):
    count = bad_spec.split(":")[1]
    with pytest.raises(
        pydantic.ValidationError,
        match=f"Invalid count: '{count}'. Must be positive integer.",
    ):
        AcceleratorSpec.model_validate(bad_spec)


@pytest.mark.parametrize("bad_spec", ["a100", "H 100", "V1OO", "A100_80GB", "gpu"])
def test_from_string_spec_unknown_accelerator_lists_available_types(bad_spec):
    with pytest.raises(pydantic.ValidationError) as exc_info:
        AcceleratorSpec.model_validate(bad_spec)

    message = str(exc_info.value)
    assert f"Unsupported accelerator type: `{bad_spec}`" in message
    for accelerator in Accelerator:
        assert accelerator.value in message


def test_from_string_spec_strips_surrounding_whitespace():
    assert AcceleratorSpec.model_validate("  H100:2  ") == AcceleratorSpec(
        accelerator=Accelerator.H100, count=2
    )


def test_from_string_spec_accepts_zero_padded_count():
    assert AcceleratorSpec.model_validate("A100:02") == AcceleratorSpec(
        accelerator=Accelerator.A100, count=2
    )


def test_underscore_prefixed_b10_member_is_parsed_by_value_not_name():
    # `Accelerator._B10` is a regular enum member whose *value* is "B10", so that is
    # the only string that parses.
    assert AcceleratorSpec.model_validate("B10") == AcceleratorSpec(
        accelerator=Accelerator._B10, count=1
    )
    assert AcceleratorSpec.model_validate("B10").to_dict() == "B10"
    with pytest.raises(pydantic.ValidationError, match="Unsupported accelerator type"):
        AcceleratorSpec.model_validate("_B10")


def test_parse_combined_spec_accepts_dict_instance_and_none():
    from_dict = AcceleratorSpec.model_validate(
        {"accelerator": Accelerator.L4, "count": 4}
    )
    assert from_dict == AcceleratorSpec(accelerator=Accelerator.L4, count=4)
    assert AcceleratorSpec.model_validate(from_dict) == from_dict
    assert AcceleratorSpec.model_validate(None) == AcceleratorSpec(
        accelerator=None, count=1
    )


def test_parse_combined_spec_instance_branch_returns_a_string():
    spec = AcceleratorSpec.model_validate("A100:2")
    # pydantic short-circuits inputs that are already instances of the model, so the
    # `isinstance(value, AcceleratorSpec)` branch is never taken during validation.
    assert AcceleratorSpec.model_validate(spec) is spec
    # Called directly, it returns a serialized *string* rather than a dict, because
    # `.dict()` goes through the plain model serializer. Harmless only because the
    # branch is unreachable, and because the string happens to re-parse.
    assert AcceleratorSpec._parse_combined_spec(spec) == "A100:2"


@pytest.mark.parametrize("bad_value", [1, 2.5, True, ["A100"], ("A100", 2)])
def test_parse_combined_spec_rejects_other_types_with_type_error(bad_value):
    # Note: `TypeError` (not `pydantic.ValidationError`) escapes the validator.
    with pytest.raises(
        TypeError, match="Expected string, dict, AcceleratorSpec, or None"
    ):
        AcceleratorSpec.model_validate(bad_value)


@pytest.mark.parametrize(
    "spec, expected",
    [
        (AcceleratorSpec(), None),
        (AcceleratorSpec(accelerator=None, count=4), None),
        (AcceleratorSpec(accelerator=Accelerator.T4, count=1), "T4"),
        (AcceleratorSpec(accelerator=Accelerator.T4, count=2), "T4:2"),
        (AcceleratorSpec(accelerator=Accelerator.H100, count=8), "H100:8"),
    ],
)
def test_to_string_spec_serialization(spec, expected):
    assert spec.model_dump() == expected
    assert spec.to_dict() == expected


@pytest.mark.parametrize("count", [1, 2, 8])
def test_string_round_trip(count):
    original = f"H100:{count}" if count > 1 else "H100"
    spec = AcceleratorSpec.model_validate(original)
    assert spec.model_dump() == original
    assert AcceleratorSpec.model_validate(spec.model_dump()) == spec


def test_json_schema_is_nullable_string():
    assert AcceleratorSpec.model_json_schema() == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    resources_schema = Resources.model_json_schema()
    assert resources_schema["$defs"]["AcceleratorSpec"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }


def test_validate_assignment_on_accelerator_spec():
    spec = AcceleratorSpec.model_validate("A100")

    spec.accelerator = Accelerator.H100
    assert spec.model_dump() == "H100"
    # Enum coercion happens on assignment too.
    spec.accelerator = "L4"  # type: ignore[assignment]
    assert spec.accelerator is Accelerator.L4

    spec.count = 4
    assert spec.model_dump() == "L4:4"

    with pytest.raises(pydantic.ValidationError):
        spec.count = -1
    with pytest.raises(pydantic.ValidationError):
        spec.accelerator = "not-a-gpu"  # type: ignore[assignment]
    # Failed assignments leave the model untouched.
    assert spec.model_dump() == "L4:4"


def test_resources_accelerator_and_use_gpu():
    assert Resources().use_gpu is False
    assert Resources().accelerator == AcceleratorSpec()

    resources = Resources(accelerator="A10G:4")
    assert resources.use_gpu is True
    assert resources.accelerator.accelerator is Accelerator.A10G
    assert resources.accelerator.count == 4
    assert resources.to_dict(verbose=True) == {
        "cpu": "1",
        "memory": "2Gi",
        "accelerator": "A10G:4",
        "use_gpu": True,
    }


def test_resources_reject_use_gpu_input_but_allow_round_trip():
    dumped = Resources(accelerator="T4:2").to_dict(verbose=True)
    assert dumped["use_gpu"] is True
    # `use_gpu` is a computed field: it is dropped on parsing rather than rejected...
    assert Resources.model_validate(dumped) == Resources(accelerator="T4:2")
    # ...and its value in the input is ignored entirely.
    assert Resources.model_validate({"accelerator": "T4:2", "use_gpu": False}).use_gpu


def test_resources_assignment_validation():
    resources = Resources()
    resources.accelerator = "H100:8"  # type: ignore[assignment]
    assert resources.accelerator == AcceleratorSpec(
        accelerator=Accelerator.H100, count=8
    )
    assert resources.use_gpu is True

    with pytest.raises(pydantic.ValidationError, match="Unsupported accelerator type"):
        resources.accelerator = "H1000"  # type: ignore[assignment]
    with pytest.raises(pydantic.ValidationError, match="Invalid cpu specification"):
        resources.cpu = "half"
    assert resources.accelerator.count == 8


@pytest.mark.parametrize("node_count, valid", [(None, True), (1, True), (2, True)])
def test_resources_node_count_valid(node_count, valid):
    resources = Resources(node_count=node_count)
    assert resources.node_count == node_count
    # `node_count` is omitted from the serialized output when unset.
    assert ("node_count" in resources.to_dict(verbose=True)) == bool(node_count)


@pytest.mark.parametrize("node_count", [0, -1, 1.5, "2", True])
def test_resources_node_count_invalid(node_count):
    with pytest.raises(pydantic.ValidationError):
        Resources(node_count=node_count)


def test_resources_instance_type_serialization():
    resources = Resources(instance_type="L4:4x16", accelerator="L4:4")
    dumped = resources.to_dict(verbose=True)
    assert dumped["instance_type"] == "L4:4x16"
    assert dumped["accelerator"] == "L4:4"
    assert Resources.model_validate(dumped) == resources
    # Omitted when unset.
    assert "instance_type" not in Resources(accelerator="L4:4").to_dict(verbose=True)


@pytest.mark.parametrize(
    "cpu_spec, valid",
    [("1", True), ("4", True), ("0.5", True), ("500m", True), ("m", True)]
    + [("500 m", False), ("m500", False), ("half", False), ("", False)],
)
def test_resources_cpu_validation(cpu_spec, valid):
    if valid:
        assert Resources(cpu=cpu_spec).cpu == cpu_spec
    else:
        with pytest.raises(pydantic.ValidationError):
            Resources(cpu=cpu_spec)


@pytest.mark.parametrize(
    "mem_spec, expected_bytes",
    [
        ("1", 1),
        ("2.5", 3),  # `math.ceil` of a plain numeric spec.
        ("512k", 512 * 10**3),
        ("2Gi", 2 * 1024**3),
        ("2G", 2 * 10**9),
        ("1Ti", 1024**4),
        ("0.5Mi", 1024**2 // 2),
        (".5Gi", 1024**3 // 2),
    ],
)
def test_resources_memory_in_bytes(mem_spec, expected_bytes):
    assert Resources(memory=mem_spec).memory_in_bytes == expected_bytes


@pytest.mark.parametrize("mem_spec", ["", "1Xi", "1 Gi", "Gi1", "-1Gi", "1e3Gi"])
def test_resources_memory_invalid(mem_spec):
    with pytest.raises(pydantic.ValidationError):
        Resources(memory=mem_spec)


def test_yaml_round_trip_preserves_accelerator(tmp_path: pathlib.Path):
    config = TrussConfig(
        resources=Resources(accelerator="H100:8", cpu="4", memory="8Gi")
    )
    path = tmp_path / "config.yaml"
    config.write_to_yaml_file(path, verbose=False)

    raw = yaml.safe_load(path.read_text())
    assert raw["resources"] == {
        "accelerator": "H100:8",
        "cpu": "4",
        "memory": "8Gi",
        "use_gpu": True,
    }
    reloaded = TrussConfig.from_yaml(path)
    assert reloaded.resources == config.resources
    assert reloaded.resources.accelerator.count == 8


def test_yaml_round_trip_without_accelerator(tmp_path: pathlib.Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"resources": {"cpu": "500m", "memory": "512Mi"}}))
    config = TrussConfig.from_yaml(path)
    assert config.resources.accelerator == AcceleratorSpec()
    assert config.resources.use_gpu is False
    assert config.to_dict(verbose=False)["resources"]["accelerator"] is None


def test_yaml_load_rejects_bad_accelerator(tmp_path: pathlib.Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"resources": {"accelerator": "H100:0"}}))
    with pytest.raises(pydantic.ValidationError, match="Invalid count"):
        TrussConfig.from_yaml(path)


# --- Tests documenting current (buggy) behavior. See PR description. -----------------


def test_bug_extra_colon_segments_are_silently_ignored():
    # BUG: only `len(parts) == 2` is handled, so any spec with more than one colon
    # silently falls back to a count of 1 instead of raising. `H100:8:0` requests a
    # single GPU rather than erroring out.
    assert AcceleratorSpec.model_validate("H100:8:0") == AcceleratorSpec(
        accelerator=Accelerator.H100, count=1
    )
    assert AcceleratorSpec.model_validate("H100:8:0").model_dump() == "H100"
    assert Resources(accelerator="H100:8:0").accelerator.count == 1


def test_bug_count_zero_drops_accelerator_but_keeps_use_gpu():
    # BUG: `count=0` is allowed by the field constraint (`ge=0`), `use_gpu` only looks
    # at `accelerator`, but the serializer emits `None` for `count <= 0`. The dumped
    # config therefore claims `use_gpu: true` with no accelerator, and the accelerator
    # is lost on round-trip.
    resources = Resources(
        accelerator=AcceleratorSpec(accelerator=Accelerator.H100, count=0)
    )
    assert resources.use_gpu is True
    dumped = resources.to_dict(verbose=True)
    assert dumped["accelerator"] is None
    assert dumped["use_gpu"] is True

    reloaded = Resources.model_validate(dumped)
    assert reloaded.accelerator.accelerator is None
    assert reloaded.use_gpu is False


@pytest.mark.parametrize("mem_spec", ["Gi", "1.2.3Gi", ".Mi"])
def test_bug_memory_accepted_but_memory_in_bytes_raises(mem_spec):
    # BUG: `_MEMORY_REGEX` allows a unit with no (or a malformed) number, so validation
    # passes, but `memory_in_bytes` then raises a bare `ValueError` from `float()`.
    resources = Resources(memory=mem_spec)
    assert resources.memory == mem_spec
    with pytest.raises(ValueError, match="could not convert string to float"):
        _ = resources.memory_in_bytes
