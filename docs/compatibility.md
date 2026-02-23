# Compatibility Policy

This document defines API/contract compatibility expectations for `AI-ToolKit-MCP`.

## Scope

Compatibility applies to:
- MCP tool names and required arguments
- HTTP envelope fields from `mcp_http_server.py`
- JSON schema files in `schemas/`
- Provenance fields used by automation (`confidence_source`, `band_source`)

## Versioning

- Tags follow SemVer (`MAJOR.MINOR.PATCH`).
- `MAJOR`: breaking contract changes.
- `MINOR`: backward-compatible additive changes.
- `PATCH`: bug fixes, no intended contract break.

## Output Contract Levels

### Stable (must not change without major bump)

- Tool names already published in `README.md`.
- Required top-level fields in schema files:
  - `schemas/get-training-observability.schema.json`
  - `schemas/compare-training-runs.schema.json`
- Provenance keys:
  - `intelligence.bottlenecks[].confidence_source`
  - `delta_to_winner.band_source.id|units|thresholds`

### Additive (allowed in minor releases)

- New optional fields in payload objects.
- New optional tools.
- New optional HTTP metadata.

### Internal (no compatibility guarantee)

- Non-schema internal helper functions.
- Internal scoring implementation details unless surfaced as stable schema fields.

## Breaking Change Rules

Any of the following is considered breaking:
- Removing or renaming an existing tool.
- Removing required schema fields.
- Changing semantic meaning of stable fields without migration path.
- Changing `band_source.id` behavior without versioned replacement.

Breaking changes require:
1. Major version bump.
2. Changelog entry with migration notes.
3. Updated schema files and tests.
4. README migration guidance.

## Deprecation Process

1. Mark field/tool as deprecated in README + changelog.
2. Keep support for at least one minor release cycle.
3. Add replacement guidance.
4. Remove only in next major release.

## Enforcement

CI explicitly validates schema contracts:
- `.github/workflows/ci.yml` runs `python -m unittest tests.test_output_schema_files`
- Full test suite also includes schema validation tests.

## Release Checklist (Contract-Sensitive)

Before tagging:
1. Run `python -m py_compile mcp_server.py mcp_http_server.py`.
2. Run `python -m unittest discover -s tests -p 'test_*.py'`.
3. Confirm schema files match emitted payloads.
4. Update `CHANGELOG.md`.
5. Tag release.
