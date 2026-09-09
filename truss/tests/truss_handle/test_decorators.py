from unittest.mock import Mock, patch

from truss.truss_handle.decorators import proxy_to_shadow_if_scattered


@proxy_to_shadow_if_scattered
def _sample_func(handle, extra):
    return (handle, extra)


def test_proxy_to_shadow_if_not_scattered():
    handle = Mock()
    handle.is_scattered.return_value = False
    assert _sample_func(handle, "arg") == (handle, "arg")


def test_proxy_to_shadow_if_scattered():
    handle = Mock()
    handle.is_scattered.return_value = True
    handle.gather.return_value = "/gathered"
    with patch("truss.truss_handle.truss_handle.TrussHandle") as mock_th:
        gathered = Mock()
        mock_th.return_value = gathered
        assert _sample_func(handle, "arg") == (gathered, "arg")
        mock_th.assert_called_once_with("/gathered")
