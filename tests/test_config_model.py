import json
from pathlib import Path
import pytest

from grok_search.config import Config, config

pytestmark = pytest.mark.usefixtures("isolated_config")


def test_default_model_is_grok_4_3_fast():
    assert Config._DEFAULT_MODEL == "grok-4.3-fast"
    cfg = Config()
    assert cfg.grok_model == "grok-4.3-fast"


def test_deprecated_model_in_env_canonicalizes_to_grok_4_3_fast(monkeypatch):
    monkeypatch.setenv("GROK_MODEL", "grok-4.20-beta")
    cfg = Config()
    assert cfg.grok_model == "grok-4.3-fast"


def test_deprecated_model_in_config_file_rewrites_on_load(tmp_path):
    config_dir = tmp_path / ".config" / "grok-search"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"model": "grok-4.20-beta"}), encoding="utf-8")

    cfg = Config()
    # Loading config should canonicalize model and rewrite the file
    assert cfg.grok_model == "grok-4.3-fast"

    saved_data = json.loads(config_file.read_text(encoding="utf-8"))
    assert saved_data.get("model") == "grok-4.3-fast"


def test_set_model_deprecated_persists_canonical(tmp_path):
    cfg = Config()
    cfg.set_model("grok-4.20-beta")
    assert cfg.grok_model == "grok-4.3-fast"

    saved_data = json.loads(cfg.config_file.read_text(encoding="utf-8"))
    assert saved_data.get("model") == "grok-4.3-fast"


def test_get_config_info_includes_canonical_model_and_deprecation_note(monkeypatch):
    monkeypatch.setenv("GROK_MODEL", "grok-4.20-beta")
    cfg = Config()
    info = cfg.get_config_info()
    assert info["remote_engine"]["model"] == "grok-4.3-fast"
    assert "grok-4.20-beta" in str(info.get("model_deprecation") or info.get("deprecated_model"))


def test_get_config_info_no_deprecation_note_for_standard_model(monkeypatch):
    monkeypatch.setenv("GROK_MODEL", "grok-4-fast")
    cfg = Config()
    info = cfg.get_config_info()
    assert info["remote_engine"]["model"] == "grok-4-fast"
    assert info.get("model_deprecation") is None and info.get("deprecated_model") is None


@pytest.mark.asyncio
async def test_switch_model_reports_deprecation_when_requested_deprecated():
    import grok_search.server as server

    res_raw = await server.switch_model("grok-4.20-beta")
    res = json.loads(res_raw)
    assert res["status"] == "✅ 成功"
    assert res["current_model"] == "grok-4.3-fast"
    assert "deprecated" in res.get("deprecation_note", "").lower() or "deprecated" in res.get("message", "").lower() or "已废弃" in str(res)
