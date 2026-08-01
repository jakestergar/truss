import pathlib
from unittest import mock

import pytest

from truss.templates.shared import secrets_resolver
from truss_chains import private_types, public_types
from truss_chains.remote_chainlet import model_skeleton


@pytest.fixture
def secrets():
    return secrets_resolver.Secrets({"key": "value"})


@pytest.fixture
def lazy_data_resolver():
    return mock.Mock()


def _config(chainlet_to_service=None):
    return {
        "model_metadata": {
            private_types.TRUSS_CONFIG_CHAINS_KEY: {
                "chainlet_to_service": chainlet_to_service or {}
            }
        }
    }


def test_init_without_dependencies_or_environment(
    tmp_path, secrets, lazy_data_resolver
):
    model = model_skeleton.TrussChainletModel(
        config=_config(),
        data_dir=tmp_path,
        secrets=secrets,
        lazy_data_resolver=lazy_data_resolver,
    )

    assert model._context.chainlet_to_service == {}
    assert model._context.data_dir == tmp_path
    assert model._context.environment is None
    assert model._context.secrets["key"] == "value"
    lazy_data_resolver.block_until_download_complete.assert_called_once_with()


def test_init_parses_environment(tmp_path, secrets, lazy_data_resolver):
    model = model_skeleton.TrussChainletModel(
        config=_config(),
        data_dir=tmp_path,
        secrets=secrets,
        lazy_data_resolver=lazy_data_resolver,
        environment={"name": "production"},
    )

    assert model._context.environment == public_types.Environment(name="production")


def test_init_populates_dependency_predict_urls(tmp_path, secrets, lazy_data_resolver):
    descriptor = public_types.DeployedServiceDescriptor(
        name="Dep",
        display_name="Dep",
        options=public_types.RPCOptions(),
        predict_url="https://example.com/predict",
    )
    with mock.patch.object(
        model_skeleton.utils,
        "populate_chainlet_service_predict_urls",
        return_value={"Dep": descriptor},
    ) as populate:
        model = model_skeleton.TrussChainletModel(
            config=_config(
                {"Dep": {"name": "Dep", "display_name": "Dep", "options": {}}}
            ),
            data_dir=pathlib.Path(tmp_path),
            secrets=secrets,
            lazy_data_resolver=lazy_data_resolver,
        )

    populate.assert_called_once()
    assert model._context.get_service_descriptor("Dep") is descriptor


def test_init_rejects_config_without_chains_metadata(
    tmp_path, secrets, lazy_data_resolver
):
    with pytest.raises(KeyError):
        model_skeleton.TrussChainletModel(
            config={"model_metadata": {}},
            data_dir=tmp_path,
            secrets=secrets,
            lazy_data_resolver=lazy_data_resolver,
        )
