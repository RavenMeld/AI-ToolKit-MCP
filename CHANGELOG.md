# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Optional HTTP auth token for `mcp_http_server.py` via `MCP_HTTP_AUTH_TOKEN`.
  - Accepted headers: `Authorization: Bearer <token>` or `X-API-Key: <token>`.

### Changed
- Normalized HTTP tool response envelope now includes:
  - `request_id`
  - `data` (parsed JSON-first payload for automation)
  - `result` (raw MCP content list retained for compatibility)

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
