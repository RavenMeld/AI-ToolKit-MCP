# ExecPlan: v0.1.1 Release Cut

## Goal
Publish `v0.1.1` including HTTP security/hardening and schema-contract upgrades.

## Scope
- Update changelog with a `0.1.1` section.
- Add release notes document.
- Update README release status.
- Run validation suite.
- Create and push annotated git tag.

## Validation
- `python -m py_compile mcp_server.py mcp_http_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`

## Result
- Release artifacts updated.
- Tag created and pushed.
