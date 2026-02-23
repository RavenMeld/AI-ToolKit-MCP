# ExecPlan: Optional Strict Mode for Proxy Header Trust

## Objective
Add an optional strict-mode toggle so forwarded client-IP headers are only allowed when a trusted proxy CIDR allowlist is configured.

## Progress
- [x] Define strict-mode behavior and compatibility constraints.
- [x] Implement strict-mode toggle in `mcp_http_server.py`.
- [x] Add tests for strict-mode validation behavior.
- [x] Update README and changelog.
- [x] Run validations.
- [x] Commit and push.

## Decisions (with rationale)
1. Keep strict mode disabled by default.
Reason: preserve backward compatibility for existing proxy-trust deployments.

2. Fail fast on invalid strict-mode configuration.
Reason: if strict mode is enabled without CIDRs, startup should fail clearly instead of silently degrading behavior.

## Validation Commands
1. `python -m py_compile mcp_server.py mcp_http_server.py`
2. `python -m unittest tests.test_http_wrapper_basic`
3. `python -m unittest discover -s tests -p 'test_*.py'`

Expected: all commands pass.

## Final Results
- Validation completed successfully:
  - `python -m py_compile mcp_server.py mcp_http_server.py`
  - `python -m unittest tests.test_http_wrapper_basic` (`Ran 26 tests ... OK`)
  - `python -m unittest discover -s tests -p 'test_*.py'` (`Ran 36 tests ... OK`)
- Strict-mode controls documented:
  - env: `MCP_HTTP_REQUIRE_TRUSTED_PROXY_CIDRS`
  - CLI: `--require-trusted-proxy-cidrs` / `--allow-any-proxy-source`
