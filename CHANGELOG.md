# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

No changes yet.

## [0.1.1] - 2026-02-23

### Added
- Optional HTTP auth token for `mcp_http_server.py` via `MCP_HTTP_AUTH_TOKEN`.
  - Accepted headers: `Authorization: Bearer <token>` or `X-API-Key: <token>`.
- HTTP hardening controls:
  - `MCP_HTTP_TOOL_TIMEOUT_SECONDS` for per-request execution timeout
  - `MCP_HTTP_MAX_BODY_BYTES` for request body size limit
- Versioned output schema files:
  - `schemas/get-training-observability.schema.json`
  - `schemas/compare-training-runs.schema.json`
- Schema validation tests:
  - `tests/test_output_schema_files.py`
- `jsonschema` dependency for schema validation.
- Compatibility policy document:
  - `docs/compatibility.md`

### Changed
- Normalized HTTP tool response envelope now includes:
  - `request_id`
  - `data` (parsed JSON-first payload for automation)
  - `result` (raw MCP content list retained for compatibility)
- HTTP wrapper returns structured error codes for hardening failures:
  - `TOOL_TIMEOUT` (`504`)
  - `PAYLOAD_TOO_LARGE` (`413`)
- CI now runs an explicit schema contract step before full unit tests.

## [0.1.0] - 2026-02-23

Initial public release for `RavenMeld/AI-ToolKit-MCP`.

### Added
- Pinned dependency manifest in `requirements.txt`.
- Minimal HTTP bridge in `mcp_http_server.py`:
  - `GET /health`
  - `GET /tools`
  - `POST /mcp/tool`
- CI workflow in `.github/workflows/ci.yml` (Python 3.10/3.11, compile + unit tests).
- HTTP wrapper tests in `tests/test_http_wrapper_basic.py`.
- Observability integration smoke test in `tests/test_get_training_observability_smoke.py`.
- Expanded API documentation in `README.md`:
  - Top 8 tool request/response examples
  - Schema excerpts for `intelligence`, `delta_to_winner`, and `alerts`
  - Provenance contract notes for `confidence_source` and `band_source`

### Quality Gates
- `python -m py_compile mcp_server.py mcp_http_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`
