# AI ToolKit MCP

AI ToolKit MCP is a Python MCP server that wraps an AI Toolkit training backend and exposes high-value training operations as MCP tools.

This repository is intentionally focused:
- `mcp_server.py` contains the MCP server and tool implementations.
- `tests/` contains contract and scoring/provenance tests.

## What You Get

- End-to-end training orchestration tools.
- Deep observability and run intelligence in one call.
- Dataset diagnostics and safe auto-improvement actions.
- Compare-run explainability with explicit provenance metadata.

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

Python packages required by `mcp_server.py` include:
- `mcp`
- `aiohttp`
- `websockets`
- `pydantic`
- `pyyaml`
- `Pillow`

Example install:

```bash
python -m pip install mcp aiohttp websockets pydantic pyyaml Pillow
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

Optional syntax check:

```bash
python -m py_compile mcp_server.py
```

## Repository Layout

- `mcp_server.py`: MCP server, tool schemas, tool handlers, intelligence/scoring helpers
- `tests/`: contract and helper tests

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
