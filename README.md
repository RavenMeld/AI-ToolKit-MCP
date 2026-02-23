# AI ToolKit MCP

AI ToolKit MCP is a Python MCP server that wraps an AI Toolkit training backend and exposes high-value training operations as MCP tools.

This repository is intentionally focused:
- `mcp_server.py` contains the MCP server and tool implementations.
- `mcp_http_server.py` provides a lightweight HTTP bridge over the same tool handlers.
- `tests/` contains contract and scoring/provenance tests.

## What You Get

- End-to-end training orchestration tools.
- Deep observability and run intelligence in one call.
- Dataset diagnostics and safe auto-improvement actions.
- Compare-run explainability with explicit provenance metadata.

## Quick Start (5 Minutes)

1. Install runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Point the MCP server at your AI Toolkit backend:

```bash
export AI_TOOLKIT_SERVER_URL=http://localhost:8675
export LOG_LEVEL=INFO
```

3. Start the MCP server:

```bash
python mcp_server.py
```

4. Optional: start the HTTP bridge (health + tool execution endpoints):

```bash
python mcp_http_server.py --host 127.0.0.1 --port 8080
```

5. Register this command in your MCP client and call `get-training-observability` with a `job_id`.

## Architecture At A Glance

1. MCP client calls a tool exposed by `mcp_server.py`.
2. The server queries AI Toolkit backend APIs for jobs, logs, metrics, and artifacts.
3. Helper layers score/aggregate speed, quality, errors, dataset, and NLP signals.
4. The server returns structured JSON contracts designed for automation, including confidence provenance fields.

## Most Used Workflows

### Validate -> Plan -> Guardrailed Run

Use these tools in order:
- `validate-training-config`
- `recommend-training-plan`
- `run-training-with-guardrails`

### One-Call Deep Diagnosis

Use:
- `get-training-observability`

Typical arguments:

```json
{
  "tool": "get-training-observability",
  "arguments": {
    "job_id": "job_12345",
    "lines": 1200,
    "include_dataset": true,
    "include_nlp": true,
    "include_evaluation": true,
    "include_next_experiment": true,
    "include_baseline_comparison": true,
    "baseline_limit": 5
  }
}
```

### Compare Multiple Runs

Use:
- `compare-training-runs`

Typical arguments:

```json
{
  "tool": "compare-training-runs",
  "arguments": {
    "job_ids": ["job_a", "job_b", "job_c"],
    "include_dataset": true,
    "include_nlp": true
  }
}
```

## Tool Surface

The server currently exposes these tools:

- `create-training-config`
- `validate-training-config`
- `recommend-training-plan`
- `run-training-with-guardrails`
- `watch-training-timeseries`
- `compare-training-runs`
- `evaluate-lora`
- `suggest-next-experiment`
- `auto-improve-dataset`
- `export-observability-report`
- `alert-routing`
- `list-configs`
- `get-config`
- `get-training-info`
- `upload-dataset`
- `list-datasets`
- `list-comfyui-models`
- `start-training`
- `get-training-status`
- `stop-training`
- `list-training-jobs`
- `export-model`
- `list-exported-models`
- `download-model`
- `get-system-stats`
- `get-training-observability`
- `get-training-logs`

## Key Contracts

### `get-training-observability`

Primary one-call intelligence endpoint. Output includes:
- speed, quality, error, dataset, and NLP analyses
- aggregate alerts
- optional evaluation and next-experiment recommendation
- optional baseline comparison
- intelligence section with decision gates and bottlenecks

Bottlenecks include confidence provenance:

```json
{
  "code": "THROUGHPUT_LOW",
  "severity": "warning",
  "message": "Observed throughput is below the adaptive healthy range for this run profile.",
  "confidence": 0.76,
  "confidence_source": "alert_category_max"
}
```

### `compare-training-runs`

Returns rank ordering plus explainability deltas against the winner.

`delta_to_winner.band_source` documents how confidence bands are derived:

```json
{
  "total": -6.4,
  "total_band": "moderate",
  "components": {
    "convergence": -8.0,
    "speed": -3.5,
    "stability": -4.2,
    "dataset": -1.0
  },
  "component_bands": {
    "convergence": "moderate",
    "speed": "moderate",
    "stability": "moderate",
    "dataset": "negligible"
  },
  "band_source": {
    "id": "absolute_delta_thresholds_v1",
    "units": "score_points",
    "thresholds": {
      "negligible_lte": 2.0,
      "moderate_lte": 8.0,
      "large_gt": 8.0
    }
  }
}
```

## Runtime Prerequisites

- Python 3.10+
- A reachable AI Toolkit server (`AI_TOOLKIT_SERVER_URL`)
- Valid filesystem mounts for dataset/output/config paths expected by the backend

Install pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Environment variables used by the server:

- `AI_TOOLKIT_SERVER_URL` (default: `http://localhost:8675`)
- `LOG_LEVEL` (default: `INFO`)
- `COMFYUI_MODEL_ROOTS` (colon-separated model roots)

Default data paths in code:
- `/ai-toolkit/datasets`
- `/ai-toolkit/outputs`
- `/ai-toolkit/configs`
- `/workspace/logs`

## Run The MCP Server

```bash
python mcp_server.py
```

The process starts an MCP stdio server (`server_name=ai-toolkit-mcp`).

## Run The HTTP Bridge

```bash
python mcp_http_server.py --host 127.0.0.1 --port 8080
```

Endpoints:
- `GET /health`
- `GET /tools`
- `POST /mcp/tool` with body: `{"name":"<tool-name>","arguments":{...}}`

## MCP Client Wiring (Example)

Example MCP command configuration:

```json
{
  "command": "python",
  "args": ["/absolute/path/to/mcp_server.py"],
  "env": {
    "AI_TOOLKIT_SERVER_URL": "http://localhost:8675",
    "LOG_LEVEL": "INFO"
  }
}
```

## Development and Validation

Run tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Current test modules:
- `tests/test_timeseries_compare_helpers.py`
- `tests/test_observability_schema_contract.py`
- `tests/test_http_wrapper_basic.py`

Optional syntax check:

```bash
python -m py_compile mcp_server.py mcp_http_server.py
```

## Repository Layout

- `mcp_server.py`: MCP server, tool schemas, tool handlers, intelligence/scoring helpers
- `mcp_http_server.py`: minimal HTTP wrapper delegating to the same MCP tool handlers
- `tests/`: contract and helper tests

## Contract Stability Notes

- `confidence_source` and `band_source` are part of the intended automation contract.
- `band_source.id` is versioned (`absolute_delta_thresholds_v1`) so downstream systems can detect scoring-method changes.
- Threshold values in `band_source.thresholds` should be consumed dynamically instead of hardcoded.

## Operational Guardrails

- Run through `run-training-with-guardrails` for long jobs instead of raw `start-training`.
- Prefer `include_baseline_comparison=true` for release/selection decisions.
- Route critical conditions using `alert-routing` to file or webhook sinks.
- Keep `AI_TOOLKIT_SERVER_URL` on trusted internal networks only.

## Troubleshooting

- `Permission denied` / backend errors:
  - Verify `AI_TOOLKIT_SERVER_URL` points to a live AI Toolkit instance.
- Missing dataset/report artifacts:
  - Ensure backend-mounted paths exist and are writable where required.
- No models found in `list-comfyui-models`:
  - Verify `COMFYUI_MODEL_ROOTS` and filesystem mounts.

## Notes

- This repo is the MCP server layer. It assumes AI Toolkit backend APIs are available and healthy.
- Provenance fields are intentional API contracts for automation consumers (`confidence_source`, `band_source`).
