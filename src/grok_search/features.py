"""Compiled-in feature registry with local-file and session-environment overrides."""

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

STAGES = ("experimental", "under-development", "stable", "deprecated", "removed")
NON_PRODUCTION_STAGES = ("experimental", "under-development")

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_ACCEPTED_TOKENS = "1/true/yes/on or 0/false/no/off"
_PRODUCTION = "production"


@dataclass(frozen=True)
class Feature:
    id: str
    key: str
    stage: str
    default_enabled: bool
    note: str = ""


REGISTRY: tuple[Feature, ...] = (
    Feature(id="tavilyEnabled", key="TAVILY_ENABLED", stage="stable", default_enabled=True),
    Feature(id="firecrawlEnabled", key="FIRECRAWL_ENABLED", stage="stable", default_enabled=True),
)

EXCEPTIONS: tuple[str, ...] = ()

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_FILE_NAME = ".features.local.json"


def get_feature(feature_id: str) -> Feature:
    for feature in REGISTRY:
        if feature.id == feature_id:
            return feature
    raise KeyError(f"Unknown feature id: {feature_id}")


def local_file_path() -> Path:
    return REPO_ROOT / LOCAL_FILE_NAME


def load_overrides(path: Path | None = None, *, keys: Iterable[str] | None = None) -> dict[str, bool]:
    target = Path(path) if path is not None else local_file_path()
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid feature overrides file: {target}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Feature overrides must be a JSON object: {target}")
    known_keys = {feature.key for feature in REGISTRY} if keys is None else set(keys)
    overrides: dict[str, bool] = {}
    for key, value in data.items():
        if key not in known_keys:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"Override for {key} must be a JSON boolean: {target}")
        overrides[key] = value
    return overrides


def _parse_env(key: str, raw: str) -> bool:
    token = str(raw).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    # Blank values resolve to disabled, matching the pre-registry default.
    if not token or token in _FALSE_TOKENS:
        return False
    raise ValueError(f"Invalid value for {key}: use {_ACCEPTED_TOKENS}")


def resolve(
    feature: Feature | str,
    *,
    environ: Mapping[str, str] | None = None,
    local_file: Path | None = None,
) -> bool:
    spec = get_feature(feature) if isinstance(feature, str) else feature
    if spec.stage == "removed":
        return False
    env = os.environ if environ is None else environ
    overrides = load_overrides(local_file, keys=(spec.key,))
    enabled = spec.default_enabled
    explicit = spec.key in overrides
    if explicit:
        enabled = overrides[spec.key]
    raw = env.get(spec.key)
    if raw is not None:
        enabled = _parse_env(spec.key, raw)
        explicit = True
    if explicit and enabled and spec.stage in NON_PRODUCTION_STAGES:
        if (env.get("GROK_SEARCH_ENV") or "").strip().lower() == _PRODUCTION:
            raise ValueError(
                f"Feature {spec.id} ({spec.key}) is {spec.stage}; "
                f"{_PRODUCTION} refuses to enable it"
            )
    return enabled


def effective_features(
    *,
    environ: Mapping[str, str] | None = None,
    local_file: Path | None = None,
) -> list[dict]:
    return [
        {
            "id": feature.id,
            "key": feature.key,
            "stage": feature.stage,
            "default": feature.default_enabled,
            "effective": resolve(feature, environ=environ, local_file=local_file),
        }
        for feature in REGISTRY
    ]


def check_invariant(
    registry: tuple[Feature, ...] = REGISTRY,
    exceptions: tuple[str, ...] = EXCEPTIONS,
) -> None:
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for feature in registry:
        if feature.stage not in STAGES:
            raise ValueError(f"Feature {feature.id} has unknown stage: {feature.stage}")
        if feature.id in seen_ids:
            raise ValueError(f"Duplicate feature id: {feature.id}")
        if feature.key in seen_keys:
            raise ValueError(f"Duplicate feature key: {feature.key}")
        seen_ids.add(feature.id)
        seen_keys.add(feature.key)
        if feature.default_enabled and feature.stage not in ("stable", "removed"):
            if feature.id not in exceptions:
                raise ValueError(
                    f"Feature {feature.id} is default-enabled at stage {feature.stage}; "
                    "only stable may default-on unless listed in exceptions"
                )
            if not feature.note.strip():
                raise ValueError(
                    f"Feature {feature.id} is an exception; a note is required"
                )
