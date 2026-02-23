# ExecPlan: AI-ToolKit-MCP Full Roadmap

## Goal
Plan and deliver all seven improvements for `RavenMeld/AI-ToolKit-MCP`:
1. top-tool request/response examples in `README.md`
2. minimal `mcp_http_server.py`
3. CI for syntax + unit tests
4. pinned `requirements.txt`
5. integration smoke test for `get-training-observability` (mocked backend)
6. output-schema docs for `intelligence`, `delta_to_winner`, `alerts`
7. `v0.1.0` release tag

## Current Baseline
- Main files: `mcp_server.py`, `README.md`, `tests/`
- Current test modules: `tests/test_timeseries_compare_helpers.py`, `tests/test_observability_schema_contract.py` (`7` tests total at baseline)
- No standalone HTTP wrapper in this split repo
- No CI workflow in this split repo
- No pinned dependency manifest in this split repo

## Success Criteria
- Users can install, run, and integrate the server from this repo alone.
- Docs include copy/paste examples and stable schema/provenance notes.
- CI blocks regressions for syntax and unit tests.
- Release `v0.1.0` is reproducible and documented.

## Scope / Non-Goals
### In scope
- Documentation, server wrapper, tests, CI, packaging, and release metadata.

### Out of scope
- Refactoring core scoring logic in `mcp_server.py`
- New model-training algorithms
- Backward-incompatible API renames

## Workstreams and Order

## Phase 0: Baseline and Branch Hygiene
### Tasks
- Create dedicated implementation branch from remote `main`.
- Capture baseline command outputs for reproducibility.

### Validation
- `python -m py_compile mcp_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`

### Exit Criteria
- Baseline passes and is recorded in plan notes.

## Phase 1: Dependency and Runtime Foundation
### Items covered
- #4 pinned `requirements.txt`

### Tasks
- Add `requirements.txt` with pinned versions for runtime/test dependencies.
- Add short install section in `README.md` aligned with pinned file.
- Confirm local install command works in clean venv.

### Validation
- `python -m pip install -r requirements.txt`
- `python -m py_compile mcp_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`

### Exit Criteria
- Fresh environment setup is deterministic.

## Phase 2: HTTP Entry Point
### Items covered
- #2 minimal `mcp_http_server.py`

### Tasks
- Add lightweight FastAPI wrapper:
  - `/health`
  - `/tools` (list tool names/docs)
  - `/mcp/tool` (execute tool by name + args)
- Reuse internal handlers from `mcp_server.py` to avoid duplicated business logic.
- Add minimal request/response models and consistent error shape.

### Validation
- `python -m py_compile mcp_http_server.py`
- Add unit test file `tests/test_http_wrapper_basic.py` for:
  - health endpoint
  - tool call success path
  - invalid tool path

### Exit Criteria
- HTTP wrapper runs standalone and passes tests.

## Phase 3: Contract and Example Docs
### Items covered
- #1 top 8 tools request/response examples
- #6 output-schema docs (`intelligence`, `delta_to_winner`, `alerts`)

### Tasks
- Add “Top 8 Tool Examples” section to `README.md`:
  - `get-training-observability`
  - `compare-training-runs`
  - `run-training-with-guardrails`
  - `evaluate-lora`
  - `suggest-next-experiment`
  - `auto-improve-dataset`
  - `export-observability-report`
  - `alert-routing`
- Add explicit JSON schema snippets for:
  - `intelligence`
  - `delta_to_winner`
  - `alerts`
- Document provenance fields and stability expectations:
  - `confidence_source`
  - `band_source`

### Validation
- Manual payload sanity check against existing tests and helper outputs.
- Ensure examples are copy/paste valid JSON.

### Exit Criteria
- README becomes integration-ready documentation, not only high-level prose.

## Phase 4: Integration Smoke Testing
### Items covered
- #5 smoke test for `get-training-observability` with mocked backend

### Tasks
- Add new test module `tests/test_get_training_observability_smoke.py`:
  - monkeypatch backend client methods
  - execute full tool path
  - assert required sections and provenance fields
- Keep test fast and deterministic (no network, no real backend).

### Validation
- `python -m unittest discover -s tests -p 'test_*.py'`

### Exit Criteria
- One explicit smoke test protects end-to-end contract path.

## Phase 5: CI
### Items covered
- #3 CI for py_compile + tests

### Tasks
- Add `.github/workflows/ci.yml`:
  - checkout
  - Python setup (3.10/3.11 matrix or single stable target)
  - install `requirements.txt`
  - run `py_compile`
  - run unit tests
- Add path filters for repo-only scope.

### Validation
- Local dry run of CI steps via shell.
- Push branch and verify GitHub Actions green.

### Exit Criteria
- PR and push checks enforce quality gate.

## Phase 6: Release Prep and Tag
### Items covered
- #7 `v0.1.0` release tag

### Tasks
- Final README pass and changelog/release notes draft.
- Verify tool list and examples match actual code.
- Create annotated tag `v0.1.0` and push tag.

### Validation
- Re-run full local validation suite.
- Confirm tag points to release-ready commit.

### Exit Criteria
- Public `v0.1.0` tag with clear release notes.

## Deliverables Map
- `requirements.txt` (new)
- `mcp_http_server.py` (new)
- `README.md` (expanded examples + schemas)
- `tests/test_http_wrapper_basic.py` (new)
- `tests/test_get_training_observability_smoke.py` (new)
- `.github/workflows/ci.yml` (new)
- `v0.1.0` tag

## Risks and Mitigations
- Risk: HTTP wrapper drifts from stdio behavior.
  - Mitigation: delegate execution to existing `handle_call_tool` and contract tests.
- Risk: Docs drift from runtime payloads.
  - Mitigation: reuse tested payload structures and add schema assertions.
- Risk: Dependency pins become stale quickly.
  - Mitigation: pick conservative stable versions and schedule post-release bump.

## Validation Checklist (Final Gate)
- `python -m py_compile mcp_server.py mcp_http_server.py`
- `python -m unittest discover -s tests -p 'test_*.py'`
- CI workflow green on default branch
- README examples validated against tool handlers
- Release tag pushed: `v0.1.0`

## Progress Log
- [x] Planning document created
- [x] Phase 0 baseline capture
- [ ] Phase 1 dependencies
- [ ] Phase 2 HTTP wrapper
- [ ] Phase 3 docs/examples/schemas
- [ ] Phase 4 smoke test
- [ ] Phase 5 CI
- [ ] Phase 6 release tag

## Phase 0 Results
- Branch state:
  - Active branch: `ai-toolkit-mcp-server-split`
  - Tracks: `ravenmeld-ai/main`
- Baseline validation:
  - `python -m py_compile mcp_server.py` -> pass
  - `python -m unittest discover -s tests -p 'test_*.py'` -> `Ran 7 tests ... OK`
- Result: baseline is green and ready for implementation Phase 1.
