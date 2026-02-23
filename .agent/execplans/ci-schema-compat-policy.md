# ExecPlan: CI Schema Gate + Compatibility Policy

## Goal
Add a clearly visible schema contract gate in CI and a written compatibility policy.

## Scope
- Update CI workflow with explicit schema-validation step.
- Add `docs/compatibility.md`.
- Link policy from README and note in changelog.

## Tasks
- [x] Add CI step: `python -m unittest tests.test_output_schema_files`
- [x] Create compatibility policy doc
- [x] Update README contract notes
- [x] Update changelog

## Validation
- `python -m py_compile mcp_server.py mcp_http_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`

## Expected Outcome
- Schema contract check is first-class in CI output.
- Compatibility rules are explicit and referenceable.

## Result
- Implemented and validated locally.
