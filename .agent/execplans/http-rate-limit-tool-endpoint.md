# ExecPlan: HTTP `/mcp/tool` Rate Limiting Hardening

## Objective
Add optional per-client rate limiting to `POST /mcp/tool` in `mcp_http_server.py`, keeping default behavior unchanged.

## Progress
- [x] Define approach and compatibility constraints.
- [x] Implement configurable in-memory rate limiter.
- [x] Add tests for 429 behavior and retry hint.
- [x] Update README and changelog.
- [x] Run validations.
- [x] Push.

## Decisions (with rationale)
1. Default disabled.
Reason: preserve existing clients and deployment behavior.

2. Per-client fixed window keyed by request remote address.
Reason: simple deterministic behavior with low runtime overhead.

3. Return structured error code `RATE_LIMITED` with retry hints.
Reason: consistent machine-readable contract for callers.

## Validation Commands
1. `python -m py_compile mcp_server.py mcp_http_server.py`
2. `python -m unittest tests.test_http_wrapper_basic`
3. `python -m unittest discover -s tests -p 'test_*.py'`

Expected: all commands pass.

## Reproduction
1. Start with rate limit enabled:
`MCP_HTTP_RATE_LIMIT_ENABLED=true MCP_HTTP_RATE_LIMIT_WINDOW_SECONDS=60 MCP_HTTP_RATE_LIMIT_MAX_REQUESTS=2 python mcp_http_server.py --host 127.0.0.1 --port 8080`

2. Send three rapid valid `POST /mcp/tool` requests from same client.
Expected: first two succeed, third returns `429` with `RATE_LIMITED`.

## Final Results
- Validation completed successfully:
  - `python -m py_compile mcp_server.py mcp_http_server.py`
  - `python -m unittest tests.test_http_wrapper_basic` (`Ran 20 tests ... OK`)
  - `python -m unittest discover -s tests -p 'test_*.py'` (`Ran 30 tests ... OK`)
- Changes pushed to `RavenMeld/AI-ToolKit-MCP` `main` in commit:
  - `dec2402` (`feat(http): add optional per-client rate limiting for /mcp/tool`)
