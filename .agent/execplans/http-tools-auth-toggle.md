# ExecPlan: HTTP `/tools` Auth Toggle Hardening

## Objective
Add an optional, backward-compatible auth requirement for `GET /tools` in `mcp_http_server.py`, while keeping default behavior unchanged.

## Progress
- [x] Inspected existing HTTP auth flow and middleware behavior.
- [x] Implemented optional `/tools` auth gate via app config.
- [x] Added CLI/env wiring for the new toggle.
- [x] Added test coverage for default-open and protected `/tools` behavior.
- [x] Updated README and changelog.
- [x] Run full validations and push.

## Decisions (with rationale)
1. Keep `/tools` open by default.
Reason: preserve compatibility for existing clients that discover tools without auth.

2. Reuse existing auth token mechanism (`Authorization: Bearer` or `X-API-Key`) for `/tools`.
Reason: avoid introducing a second credential path and reduce configuration complexity.

3. Add explicit CLI overrides (`--auth-for-tools` / `--no-auth-for-tools`) with env-backed default.
Reason: predictable operational control in local and deployment scripts.

## Validation Commands
1. `python -m py_compile mcp_server.py mcp_http_server.py`
Expected: no output, zero exit.

2. `python -m unittest tests.test_http_wrapper_basic`
Expected: all tests pass, including new `/tools` auth tests.

3. `python -m unittest discover -s tests -p 'test_*.py'`
Expected: full suite passes.

## Reproduction
1. Start server with default behavior:
`python mcp_http_server.py --host 127.0.0.1 --port 8080`

2. Verify `/tools` is open:
`curl -s http://127.0.0.1:8080/tools`

3. Start server with protected `/tools`:
`MCP_HTTP_AUTH_TOKEN=change-me MCP_HTTP_AUTH_FOR_TOOLS=true python mcp_http_server.py --host 127.0.0.1 --port 8080`

4. Verify `/tools` unauthorized without token, then authorized with token:
`curl -i http://127.0.0.1:8080/tools`
`curl -i -H 'X-API-Key: change-me' http://127.0.0.1:8080/tools`

## Final Results
- Validation completed successfully:
  - `python -m py_compile mcp_server.py mcp_http_server.py`
  - `python -m unittest tests.test_http_wrapper_basic` (`Ran 15 tests ... OK`)
  - `python -m unittest discover -s tests -p 'test_*.py'` (`Ran 25 tests ... OK`)
- Changes pushed to `RavenMeld/AI-ToolKit-MCP` `main` in commit:
  - `38c02fc` (`feat(http): add optional auth for /tools endpoint`)
