# ExecPlan: Trusted-Proxy CIDR Allowlist for Rate-Limit Header Trust

## Objective
Add an optional CIDR allowlist so forwarded client-IP headers are only trusted for `POST /mcp/tool` rate limiting when the direct upstream address is explicitly trusted.

## Progress
- [x] Define compatibility behavior and security constraints.
- [x] Implement CIDR parsing/validation and allowlist enforcement in `mcp_http_server.py`.
- [x] Add tests for allowlisted vs non-allowlisted proxy-source behavior and invalid CIDR validation.
- [x] Update README and changelog.
- [x] Run validations.
- [x] Commit and push.

## Decisions (with rationale)
1. Keep trust-proxy toggle behavior backward compatible when allowlist is empty.
Reason: existing deployments that already enabled proxy trust should not break unexpectedly.

2. Enforce allowlist only when `MCP_HTTP_TRUST_PROXY_FOR_RATE_LIMIT=true` (or equivalent CLI flag) is enabled.
Reason: direct deployments that do not trust forwarded headers should remain unaffected.

3. Validate CIDRs at startup/app construction and fail fast on invalid entries.
Reason: avoids silent misconfiguration that could weaken rate-limit identity controls.

## Validation Commands
1. `python -m py_compile mcp_server.py mcp_http_server.py`
2. `python -m unittest tests.test_http_wrapper_basic`
3. `python -m unittest discover -s tests -p 'test_*.py'`

Expected: all commands pass.

## Reproduction
1. Start with trust enabled but allowlist excluding local source:
`MCP_HTTP_RATE_LIMIT_ENABLED=true MCP_HTTP_RATE_LIMIT_MAX_REQUESTS=1 MCP_HTTP_TRUST_PROXY_FOR_RATE_LIMIT=true MCP_HTTP_TRUSTED_PROXY_CIDRS=10.0.0.0/8 python mcp_http_server.py --host 127.0.0.1 --port 8080`

2. Send requests with different `X-Forwarded-For` values.
Expected: forwarded header ignored, second request rate limited as same client.

3. Start with local source allowlisted:
`MCP_HTTP_RATE_LIMIT_ENABLED=true MCP_HTTP_RATE_LIMIT_MAX_REQUESTS=1 MCP_HTTP_TRUST_PROXY_FOR_RATE_LIMIT=true MCP_HTTP_TRUSTED_PROXY_CIDRS=127.0.0.0/8,::1/128 python mcp_http_server.py --host 127.0.0.1 --port 8080`

4. Send requests with different `X-Forwarded-For` values.
Expected: distinct forwarded client identities respected for allowlisted source.

## Final Results
- Validation completed successfully:
  - `python -m py_compile mcp_server.py mcp_http_server.py`
  - `python -m unittest tests.test_http_wrapper_basic` (`Ran 25 tests ... OK`)
  - `python -m unittest discover -s tests -p 'test_*.py'` (`Ran 35 tests ... OK`)
- CIDR allowlist controls documented in README + changelog:
  - env: `MCP_HTTP_TRUSTED_PROXY_CIDRS`
  - CLI: `--trusted-proxy-cidrs`
