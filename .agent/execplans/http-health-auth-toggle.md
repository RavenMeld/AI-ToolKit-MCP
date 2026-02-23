# ExecPlan: HTTP `/health` Auth Toggle Hardening

## Objective
Add an optional, backward-compatible auth requirement for `GET /health` in `mcp_http_server.py`, while keeping default behavior unchanged.

## Progress
- [x] Inspected current `/health` and auth handling.
- [ ] Implement optional `/health` auth gate via app config.
- [ ] Add CLI/env wiring for the new toggle.
- [ ] Add test coverage for default-open and protected `/health` behavior.
- [ ] Update README and changelog.
- [x] Implement optional `/health` auth gate via app config.
- [x] Add CLI/env wiring for the new toggle.
- [x] Add test coverage for default-open and protected `/health` behavior.
- [x] Update README and changelog.
- [x] Run validations and push.

## Decisions (with rationale)
1. Keep `/health` open by default.
Reason: preserve compatibility for probes that do not include auth headers.

2. Reuse existing auth token mechanism (`Authorization: Bearer` or `X-API-Key`).
Reason: one auth mechanism across endpoints avoids operational drift.

3. Add explicit CLI overrides (`--auth-for-health` / `--no-auth-for-health`) with env-backed default.
Reason: deterministic runtime control in deployment scripts.

## Validation Commands
1. `python -m py_compile mcp_server.py mcp_http_server.py`
2. `python -m unittest tests.test_http_wrapper_basic`
3. `python -m unittest discover -s tests -p 'test_*.py'`

Expected: all commands succeed with zero failures.

## Reproduction
1. Default behavior (open health):
`python mcp_http_server.py --host 127.0.0.1 --port 8080`

2. Protected health:
`MCP_HTTP_AUTH_TOKEN=change-me MCP_HTTP_AUTH_FOR_HEALTH=true python mcp_http_server.py --host 127.0.0.1 --port 8080`

3. Verify unauthorized then authorized:
`curl -i http://127.0.0.1:8080/health`
`curl -i -H 'X-API-Key: change-me' http://127.0.0.1:8080/health`

## Final Results
- Validation completed successfully:
  - `python -m py_compile mcp_server.py mcp_http_server.py`
  - `python -m unittest tests.test_http_wrapper_basic` (`Ran 18 tests ... OK`)
  - `python -m unittest discover -s tests -p 'test_*.py'` (`Ran 28 tests ... OK`)
- Changes pushed to `RavenMeld/AI-ToolKit-MCP` `main` in commit:
  - `37cee83` (`feat(http): add optional auth for /health endpoint`)
