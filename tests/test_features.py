import json

import pytest

from grok_search import features
from grok_search.config import Config
from grok_search.features import (
    REGISTRY,
    Feature,
    check_invariant,
    effective_features,
    get_feature,
    load_overrides,
    local_file_path,
    resolve,
)

pytestmark = pytest.mark.usefixtures("isolated_config")

EXPERIMENTAL = Feature(id="betaFlag", key="BETA_FLAG", stage="experimental", default_enabled=False)
UNDER_DEVELOPMENT = Feature(id="devFlag", key="DEV_FLAG", stage="under-development", default_enabled=False)
REMOVED = Feature(id="legacyFlag", key="LEGACY_FLAG", stage="removed", default_enabled=False)


@pytest.fixture
def missing_local_file(tmp_path):
    return tmp_path / "absent.json"


def _write_local_file(tmp_path, payload):
    path = tmp_path / "local.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# Invariants


def test_registry_invariant_holds_and_identifiers_are_unique():
    assert check_invariant() is None
    ids = [feature.id for feature in REGISTRY]
    keys = [feature.key for feature in REGISTRY]
    assert len(ids) == len(set(ids))
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "registry",
    [
        (Feature(id="expFlag", key="EXP_FLAG", stage="experimental", default_enabled=True),),
        (Feature(id="devFlag", key="DEV_FLAG", stage="under-development", default_enabled=True),),
        (
            Feature(id="dupId", key="FIRST_KEY", stage="stable", default_enabled=False),
            Feature(id="dupId", key="SECOND_KEY", stage="stable", default_enabled=False),
        ),
        (
            Feature(id="firstId", key="SAME_KEY", stage="stable", default_enabled=False),
            Feature(id="secondId", key="SAME_KEY", stage="stable", default_enabled=False),
        ),
        (Feature(id="oddFlag", key="ODD_FLAG", stage="beta", default_enabled=False),),
    ],
    ids=[
        "default-on-experimental",
        "default-on-under-development",
        "duplicate-id",
        "duplicate-key",
        "unknown-stage",
    ],
)
def test_check_invariant_rejects_invalid_registries(registry):
    with pytest.raises(ValueError):
        check_invariant(registry=registry)


def test_check_invariant_accepts_exception_for_default_on_non_stable():
    feature = Feature(
        id="expFlag",
        key="EXP_FLAG",
        stage="experimental",
        default_enabled=True,
        note="dogfooded on the operator host",
    )
    with pytest.raises(ValueError):
        check_invariant(registry=(feature,))
    assert check_invariant(registry=(feature,), exceptions=("expFlag",)) is None


def test_check_invariant_requires_note_for_exceptions():
    feature = Feature(id="expFlag", key="EXP_FLAG", stage="experimental", default_enabled=True)
    with pytest.raises(ValueError, match="note"):
        check_invariant(registry=(feature,), exceptions=("expFlag",))


def test_check_invariant_ignores_removed_defaults():
    removed_default_on = Feature(id="gone", key="GONE", stage="removed", default_enabled=True)
    assert check_invariant(registry=(removed_default_on,)) is None


def test_unknown_feature_id_raises():
    with pytest.raises(KeyError):
        get_feature("notAFeature")


# Precedence


def test_compiled_default_applies_without_overrides(missing_local_file):
    assert resolve("tavilyEnabled", environ={}, local_file=missing_local_file) is True
    assert resolve("firecrawlEnabled", environ={}, local_file=missing_local_file) is True


def test_local_file_overrides_compiled_default(tmp_path):
    path = _write_local_file(tmp_path, {"TAVILY_ENABLED": False})
    assert resolve("tavilyEnabled", environ={}, local_file=path) is False
    assert resolve("firecrawlEnabled", environ={}, local_file=path) is True


@pytest.mark.parametrize(
    ("file_value", "env_value", "expected"),
    [(False, "1", True), (True, "0", False)],
    ids=["env-enables-over-file-disable", "env-disables-over-file-enable"],
)
def test_env_beats_local_file_in_both_directions(tmp_path, file_value, env_value, expected):
    path = _write_local_file(tmp_path, {"TAVILY_ENABLED": file_value})
    assert resolve("tavilyEnabled", environ={"TAVILY_ENABLED": env_value}, local_file=path) is expected


def test_missing_local_file_behaves_as_no_overrides(missing_local_file):
    assert load_overrides(missing_local_file) == {}
    assert resolve("tavilyEnabled", environ={"TAVILY_ENABLED": "0"}, local_file=missing_local_file) is False


def test_local_file_path_follows_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr(features, "REPO_ROOT", tmp_path)
    assert local_file_path() == tmp_path / features.LOCAL_FILE_NAME


# Local file validation


def test_invalid_local_json_names_the_file(tmp_path):
    path = tmp_path / "local.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_overrides(path)
    assert str(path) in str(excinfo.value)


def test_non_object_local_json_names_the_file(tmp_path):
    path = _write_local_file(tmp_path, ["TAVILY_ENABLED"])
    with pytest.raises(ValueError) as excinfo:
        resolve("tavilyEnabled", environ={}, local_file=path)
    assert str(path) in str(excinfo.value)


def test_non_boolean_local_value_names_file_and_key(tmp_path):
    path = _write_local_file(tmp_path, {"TAVILY_ENABLED": "false"})
    with pytest.raises(ValueError) as excinfo:
        load_overrides(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "TAVILY_ENABLED" in message


def test_unknown_local_keys_are_ignored(tmp_path):
    path = _write_local_file(tmp_path, {"UNKNOWN_FLAG": True})
    assert load_overrides(path) == {}


# Environment parsing


def test_invalid_env_value_names_key_and_hides_raw_value(missing_local_file):
    with pytest.raises(ValueError) as excinfo:
        resolve("tavilyEnabled", environ={"TAVILY_ENABLED": "maybe"}, local_file=missing_local_file)
    message = str(excinfo.value)
    assert "TAVILY_ENABLED" in message
    assert "maybe" not in message


@pytest.mark.parametrize("raw", ["", "   "], ids=["empty", "whitespace"])
def test_blank_env_value_resolves_disabled(missing_local_file, raw):
    assert resolve("tavilyEnabled", environ={"TAVILY_ENABLED": raw}, local_file=missing_local_file) is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", "  On  "])
def test_env_true_tokens(missing_local_file, raw):
    assert resolve("tavilyEnabled", environ={"TAVILY_ENABLED": raw}, local_file=missing_local_file) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE", " Off "])
def test_env_false_tokens(missing_local_file, raw):
    assert resolve("firecrawlEnabled", environ={"FIRECRAWL_ENABLED": raw}, local_file=missing_local_file) is False


# Removed stage


def test_removed_feature_is_inert_for_env(monkeypatch, missing_local_file):
    monkeypatch.setattr(features, "REGISTRY", REGISTRY + (REMOVED,))
    assert resolve("legacyFlag", environ={"LEGACY_FLAG": "true"}, local_file=missing_local_file) is False
    assert resolve("legacyFlag", environ={"LEGACY_FLAG": "junk"}, local_file=missing_local_file) is False


def test_removed_feature_is_inert_for_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(features, "REGISTRY", REGISTRY + (REMOVED,))
    path = _write_local_file(tmp_path, {"LEGACY_FLAG": True})
    assert load_overrides(path) == {"LEGACY_FLAG": True}
    assert resolve("legacyFlag", environ={}, local_file=path) is False


# Production rule


@pytest.mark.parametrize("feature", [EXPERIMENTAL, UNDER_DEVELOPMENT], ids=["experimental", "under-development"])
def test_production_refuses_env_enable_of_non_stable_feature(monkeypatch, missing_local_file, feature):
    monkeypatch.setattr(features, "REGISTRY", REGISTRY + (feature,))
    environ = {"GROK_SEARCH_ENV": "production", feature.key: "1"}
    with pytest.raises(ValueError) as excinfo:
        resolve(feature.id, environ=environ, local_file=missing_local_file)
    message = str(excinfo.value)
    assert feature.id in message
    assert feature.key in message
    assert feature.stage in message


def test_production_refuses_local_file_enable_of_non_stable_feature(monkeypatch, tmp_path):
    monkeypatch.setattr(features, "REGISTRY", REGISTRY + (EXPERIMENTAL,))
    path = _write_local_file(tmp_path, {"BETA_FLAG": True})
    with pytest.raises(ValueError):
        resolve("betaFlag", environ={"GROK_SEARCH_ENV": "production"}, local_file=path)


def test_production_allows_explicit_disable_of_non_stable_feature(monkeypatch, missing_local_file):
    monkeypatch.setattr(features, "REGISTRY", REGISTRY + (EXPERIMENTAL,))
    environ = {"GROK_SEARCH_ENV": "production", "BETA_FLAG": "0"}
    assert resolve("betaFlag", environ=environ, local_file=missing_local_file) is False


def test_production_allows_enabling_stable_feature(missing_local_file):
    environ = {"GROK_SEARCH_ENV": "  PRODUCTION  ", "TAVILY_ENABLED": "1"}
    assert resolve("tavilyEnabled", environ=environ, local_file=missing_local_file) is True


@pytest.mark.parametrize("environ", [{}, {"GROK_SEARCH_ENV": "development"}], ids=["unset", "development"])
def test_development_allows_enabling_non_stable_feature(monkeypatch, missing_local_file, environ):
    monkeypatch.setattr(features, "REGISTRY", REGISTRY + (EXPERIMENTAL,))
    env = dict(environ, BETA_FLAG="1")
    assert resolve("betaFlag", environ=env, local_file=missing_local_file) is True


# Effective surface


def test_effective_features_reports_registry_shape(missing_local_file):
    entries = effective_features(environ={}, local_file=missing_local_file)
    assert [entry["id"] for entry in entries] == [feature.id for feature in REGISTRY]
    for entry in entries:
        assert set(entry) == {"id", "key", "stage", "default", "effective"}
        assert isinstance(entry["default"], bool)
        assert isinstance(entry["effective"], bool)


def test_empty_config_resolves_both_real_flags_enabled(missing_local_file):
    assert resolve("tavilyEnabled", environ={}, local_file=missing_local_file) is True
    assert resolve("firecrawlEnabled", environ={}, local_file=missing_local_file) is True
    entries = effective_features(environ={}, local_file=missing_local_file)
    assert [entry["effective"] for entry in entries] == [True, True]


# Config surface


@pytest.mark.parametrize(
    ("flag_key", "attribute"),
    [("TAVILY_ENABLED", "tavily_enabled"), ("FIRECRAWL_ENABLED", "firecrawl_enabled")],
)
@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("true", True), ("0", False), ("false", False)])
def test_config_flag_surface_honours_env(monkeypatch, tmp_path, flag_key, attribute, raw, expected):
    monkeypatch.setattr(features, "REPO_ROOT", tmp_path)
    monkeypatch.setenv(flag_key, raw)
    assert getattr(Config(), attribute) is expected


def test_config_surface_reads_repo_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(features, "REPO_ROOT", tmp_path)
    (tmp_path / features.LOCAL_FILE_NAME).write_text(
        json.dumps({"TAVILY_ENABLED": False}), encoding="utf-8"
    )
    cfg = Config()
    assert cfg.tavily_enabled is False
    assert cfg.firecrawl_enabled is True


def test_production_blocks_non_stable_flag_through_config_surface(monkeypatch, tmp_path):
    experimental = Feature(
        id="tavilyEnabled", key="TAVILY_ENABLED", stage="experimental", default_enabled=False
    )
    registry = (experimental,) + tuple(feature for feature in REGISTRY if feature.id != "tavilyEnabled")
    monkeypatch.setattr(features, "REGISTRY", registry)
    monkeypatch.setattr(features, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GROK_SEARCH_ENV", "production")
    monkeypatch.setenv("TAVILY_ENABLED", "1")
    with pytest.raises(ValueError) as excinfo:
        Config().tavily_enabled
    message = str(excinfo.value)
    assert "tavilyEnabled" in message
    assert "TAVILY_ENABLED" in message
    assert "experimental" in message
