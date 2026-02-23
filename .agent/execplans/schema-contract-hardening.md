# ExecPlan: Schema Contract Hardening

## Goal
Add strict machine-readable output schemas for core observability/comparison tools and enforce them in tests.

## Scope
- Add JSON Schema files for:
  - `get-training-observability`
  - `compare-training-runs`
- Add schema-validation unit tests that execute tool handlers and validate payloads.
- Document schema locations in README and changelog.
- Keep compatibility with current output contracts.

## Decisions
- Use JSON Schema Draft 2020-12.
- Validate using `jsonschema` Python library in unit tests.
- Keep schemas strict at top-level (`additionalProperties: false`) with required keys, while allowing nested extensibility where outputs are intentionally richer.

## Work Items
- [x] Add `jsonschema` to `requirements.txt`.
- [x] Add `schemas/get-training-observability.schema.json`.
- [x] Add `schemas/compare-training-runs.schema.json`.
- [x] Add `tests/test_output_schema_files.py`.
- [x] Update `README.md` and `CHANGELOG.md`.

## Validation Commands
- `python -m pip install -r requirements.txt --disable-pip-version-check`
- `python -m py_compile mcp_server.py mcp_http_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`

## Expected Results
- New schema files are present and JSON-valid.
- Unit test run includes schema validation tests and passes.
- Documentation references schema file paths.

## Results
- Added all targeted schema and test files.
- Test suite passes with schema validation included.
