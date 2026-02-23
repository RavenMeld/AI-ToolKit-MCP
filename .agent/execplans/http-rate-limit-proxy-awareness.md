# ExecPlan: Proxy-Aware Client Keying for HTTP Rate Limiting

## Objective
Add optional proxy-aware client identity extraction for `POST /mcp/tool` rate limiting, with secure-by-default behavior (proxy headers not trusted unless enabled).

## Progress
- [x] Define compatibility and security constraints.
- [x] Implement optional trust-proxy controls in `mcp_http_server.py`.
- [x] Add tests for trusted vs untrusted proxy header behavior.
- [x] Update README and changelog.
- [x] Run validations.
- [x] Push.

## Decisions (with rationale)
1. Default: do not trust proxy headers.
Reason: secure default avoids spoofing in direct-exposed deployments.

2. Optional trust toggle with explicit header name.
Reason: supports reverse proxy deployments without forcing one header convention.

3. Use first value from comma-separated forwarded chain.
Reason: common de-facto behavior for client IP extraction from proxy chains.

## Validation Commands
1. `python -m py_compile mcp_server.py mcp_http_server.py`
2. `python -m unittest tests.test_http_wrapper_basic`
3. `python -m unittest discover -s tests -p 'test_*.py'`

Expected: all commands pass.

## Reproduction
1. Start with rate limiting and proxy trust disabled:
`MCP_HTTP_RATE_LIMIT_ENABLED=true MCP_HTTP_RATE_LIMIT_MAX_REQUESTS=1 python mcp_http_server.py --host 127.0.0.1 --port 8080`

2. Send two requests with different `X-Forwarded-For` values.
Expected: second request still rate limited (same direct remote).

3. Start with proxy trust enabled:
`MCP_HTTP_RATE_LIMIT_ENABLED=true MCP_HTTP_RATE_LIMIT_MAX_REQUESTS=1 MCP_HTTP_TRUST_PROXY_FOR_RATE_LIMIT=true python mcp_http_server.py --host 127.0.0.1 --port 8080`

4. Send two requests with different forwarded client IPs.
Expected: both succeed (distinct client keys).

## Final Results
- Validation completed successfully:
  - `python -m py_compile mcp_server.py mcp_http_server.py`
  - `python -m unittest tests.test_http_wrapper_basic` (`Ran 22 tests ... OK`)
  - `python -m unittest discover -s tests -p 'test_*.py'` (`Ran 32 tests ... OK`)
- Changes pushed to `RavenMeld/AI-ToolKit-MCP` `main` in commit:
  - `72f6144` (`feat(http): add optional proxy-aware rate-limit client identity`)
