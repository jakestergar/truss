import datetime
import os
import subprocess
import sys
from unittest import mock

import packaging.version

from truss.util.user_config import (
    AppSettings,
    FeatureFlags,
    Preferences,
    State,
    UpdateInfo,
    VersionInfo,
    _has_defaulted_fields,
    _SettingsWrapper,
    _StateWrapper,
    _strip_none,
    _truss_is_git_branch,
    _update_toml_document,
)


def test_import_truss_does_not_create_config_dir(tmp_path):
    """Regression test: `import truss` must not create ~/.config/truss as a side effect."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, os; "
                "import truss; "
                "config_dir = pathlib.Path(os.environ['XDG_CONFIG_HOME']) / 'truss'; "
                "assert not config_dir.exists(), "
                "f'{config_dir} was created as an import side effect'"
            ),
        ],
        env={**os.environ, "XDG_CONFIG_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_strip_none():
    assert _strip_none({"a": 1, "b": None, "c": {"d": 2, "e": None}}) == {
        "a": 1,
        "c": {"d": 2},
    }


def test_has_defaulted_fields_with_defaults():
    settings = AppSettings()
    assert _has_defaulted_fields(settings) is True


def test_has_defaulted_fields_explicit():
    settings = AppSettings(
        preferences=Preferences(
            include_git_info=True,
            auto_upgrade_command_template="cmd",
            check_for_updates=False,
        ),
        feature_flags=FeatureFlags(enable_auto_upgrade=True),
    )
    assert _has_defaulted_fields(settings) is False


def test_update_toml_document(tmp_path):
    import tomlkit

    doc = tomlkit.document()
    doc["foo"] = 1
    _update_toml_document(doc, {"foo": 2, "bar": None, "baz": "qux"})
    assert doc["foo"] == 2
    assert "bar" not in doc
    assert doc["baz"] == "qux"


def test_settings_wrapper_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    wrapper = _SettingsWrapper.read_or_create()
    assert wrapper.include_git_info is False
    wrapper.auto_upgrade_command_template = "echo upgrade"
    wrapper2 = _SettingsWrapper.read_or_create()
    assert wrapper2.auto_upgrade_command_template == "echo upgrade"


def test_settings_wrapper_reads_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config_dir = tmp_path / "truss"
    config_dir.mkdir(parents=True)
    (config_dir / "settings.toml").write_text(
        "[preferences]\ninclude_git_info = true\ncheck_for_updates = false\n",
        encoding="utf-8",
    )
    wrapper = _SettingsWrapper.read_or_create()
    assert wrapper.include_git_info is True
    assert wrapper.check_for_updates is False


def test_truss_is_git_branch_no_git(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert _truss_is_git_branch() is False


def test_version_info_parsing():
    info = VersionInfo.model_validate(
        {
            "latest_version": "0.18.0",
            "yanked_versions": ["0.17.0"],
            "last_check": datetime.datetime.now().isoformat(),
        }
    )
    assert info.latest_version == packaging.version.Version("0.18.0")
    assert packaging.version.Version("0.17.0") in info.yanked_versions


def test_state_wrapper_should_check_for_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    state = State()
    wrapper = _StateWrapper(state)
    assert wrapper._should_check_for_updates() is True
    state.version_info.last_check = datetime.datetime.now()
    assert wrapper._should_check_for_updates() is False


@mock.patch("truss.util.user_config.requests.get")
def test_state_wrapper_update_version_info(mock_get, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    mock_get.return_value.json.return_value = {
        "info": {"version": "0.18.0"},
        "releases": {"0.17.0": [{"yanked": True}], "0.18.0": [{"yanked": False}]},
    }
    state = State()
    wrapper = _StateWrapper(state)
    wrapper._update_version_info()
    assert wrapper._state.version_info.latest_version == packaging.version.Version(
        "0.18.0"
    )


@mock.patch("truss.util.user_config._truss_is_git_branch", return_value=False)
@mock.patch("truss.util.user_config._StateWrapper._should_check_for_updates")
@mock.patch("truss.util.user_config._StateWrapper._update_version_info")
def test_should_upgrade_outdated(
    mock_update, mock_should_check, mock_git, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    mock_should_check.return_value = False
    state = State(
        version_info=VersionInfo(latest_version=packaging.version.Version("0.18.0"))
    )
    wrapper = _StateWrapper(state)
    info = wrapper.should_upgrade("0.17.0")
    assert info.upgrade_recommended is True
    assert info.latest_version == "0.18.0"


@mock.patch("truss.util.user_config._truss_is_git_branch", return_value=True)
@mock.patch("truss.util.user_config._StateWrapper._should_check_for_updates")
@mock.patch("truss.util.user_config._StateWrapper._update_version_info")
def test_should_upgrade_skips_in_git_repo(
    mock_update, mock_should_check, mock_git, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    mock_should_check.return_value = False
    state = State(
        version_info=VersionInfo(latest_version=packaging.version.Version("0.18.0"))
    )
    wrapper = _StateWrapper(state)
    info = wrapper.should_upgrade("0.17.0")
    assert info.upgrade_recommended is False


def test_should_notify_upgrade_no_update(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    state = State(
        version_info=VersionInfo(
            latest_version=packaging.version.Version("0.18.0"),
            last_check=datetime.datetime.now(),
        )
    )
    wrapper = _StateWrapper(state)
    assert wrapper.should_notify_upgrade("0.18.0") is None


def test_update_info_model():
    info = UpdateInfo(
        upgrade_recommended=True, reason="outdated", latest_version="0.18.0"
    )
    assert info.upgrade_recommended is True
