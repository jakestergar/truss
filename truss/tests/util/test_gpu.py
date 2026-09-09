from unittest import mock

from truss.util.gpu import get_gpu_memory


def test_get_gpu_memory_success():
    output = "memory.used [MiB]\n1234 MiB\n"
    with mock.patch(
        "truss.util.gpu.sp.check_output", return_value=output.encode("ascii")
    ):
        assert get_gpu_memory() == 1234


def test_get_gpu_memory_file_not_found():
    with mock.patch("truss.util.gpu.sp.check_output", side_effect=FileNotFoundError()):
        assert get_gpu_memory() is None
