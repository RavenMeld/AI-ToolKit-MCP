# Release Notes: v0.1.1

Release date: 2026-02-23

## Highlights

- HTTP auth guard for tool execution in `mcp_http_server.py`:
  - `MCP_HTTP_AUTH_TOKEN`
  - `Authorization: Bearer <token>` or `X-API-Key: <token>`
- Normalized HTTP response envelope:
  - `request_id`
  - `data` (JSON-first parsed payload for automation clients)
  - `result` (raw MCP content, retained for compatibility)
- HTTP hardening controls:
  - `MCP_HTTP_TOOL_TIMEOUT_SECONDS` / `--tool-timeout-seconds`
  - `MCP_HTTP_MAX_BODY_BYTES` / `--max-body-bytes`
  - Error codes: `TOOL_TIMEOUT` (`504`), `PAYLOAD_TOO_LARGE` (`413`)
- Versioned machine-readable output schemas:
  - `schemas/get-training-observability.schema.json`
  - `schemas/compare-training-runs.schema.json`
- Schema contract enforcement in CI:
  - explicit step for `tests/test_output_schema_files.py`
- Compatibility policy documentation:
  - `docs/compatibility.md`

## Validation Snapshot

- `python -m py_compile mcp_server.py mcp_http_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`

## Upgrade Notes

- `v0.1.1` is additive and intended to be backward compatible.
- Existing consumers reading `result` remain supported.
- New consumers should prefer `data` for structured automation paths.
