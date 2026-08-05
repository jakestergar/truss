from unittest import mock

from truss.util import gpu

NVIDIA_SMI_OUTPUT = b"memory.used [MiB]\n1024 MiB\n"


def test_get_gpu_memory_parses_nvidia_smi_output():
    with mock.patch.object(
        gpu.sp, "check_output", return_value=NVIDIA_SMI_OUTPUT
    ) as check_output:
        assert gpu.get_gpu_memory() == 1024

    check_output.assert_called_once_with(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]
    )


def test_get_gpu_memory_returns_none_without_nvidia_smi():
    with mock.patch.object(gpu.sp, "check_output", side_effect=FileNotFoundError):
        assert gpu.get_gpu_memory() is None
