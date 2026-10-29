# Feature Flags

Flags are compiled into `src/grok_search/features.py`; there is no runtime registry service.
Registry edits are code changes, so they ship and get reviewed with the package.

## Stages

| Stage | Meaning |
| --- | --- |
| `experimental` | Prototype; may change or disappear. Not production-safe. |
| `under-development` | Incomplete behaviour. Not production-safe. |
| `stable` | Supported; default-on is allowed. |
| `deprecated` | Still resolves, scheduled for removal. |
| `removed` | Inert: resolves `False`, never raises, default ignored. |

## Precedence

Compiled `default_enabled` < repo-root `.features.local.json` (git-ignored, flat JSON keyed
by flag key) < the session environment variable named by the flag's `key`.

## Values

`1/true/yes/on` enable and `0/false/no/off` disable, case-insensitive with whitespace
stripped. A blank value resolves to disabled, matching the pre-registry behaviour. Any other
value raises `ValueError` naming the variable without echoing the value.

## Promotion and invariant

Flip `stage` to `stable` only after the flag has shipped default-on and been reviewed.
`check_invariant()` rejects an unknown stage, a duplicate id, a duplicate key, and a
`default_enabled=True` flag outside `stable`/`removed` unless listed in `EXCEPTIONS`.

## Production rule

With `GROK_SEARCH_ENV=production`, explicitly enabling an `experimental` or
`under-development` flag via env or local file raises `ValueError` naming the flag id, key,
and stage. Explicit disable is allowed; an unset or other value keeps development behaviour.

## Running the two legs

    TAVILY_ENABLED=0 uv run --frozen --extra dev pytest -q
    TAVILY_ENABLED=1 uv run --frozen --extra dev pytest -q
    FIRECRAWL_ENABLED=0 uv run --frozen --extra dev pytest -q
    FIRECRAWL_ENABLED=1 uv run --frozen --extra dev pytest -q

Proposal: `wiki projects/grok-search/architecture/2026-10-28-ff01-feature-flag-proposal.md`.
