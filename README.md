# AI Toolkit MCP Server (local diffusion AI model trainer)
###  Warning: Requires a powerful GPU!
A containerized [AI Toolkit](https://github.com/ostris/ai-toolkit) setup with MCP (Model Context Protocol) integration for training LoRA models and fine-tuning diffusion models. This provides a complete solution for training custom LoRA models with full MCP integration, allowing AI assistants to manage the entire training workflow.

## Usage

See the [template repository](https://github.com/AndrewAltimit/template-repo) for a complete example. Also includes the [ComfyUI MCP Server](https://gist.github.com/AndrewAltimit/f2a21b1a075cc8c9a151483f89e0f11e) used for creating images/videos from the trained models.

![mcp-demo](https://raw.githubusercontent.com/AndrewAltimit/template-repo/refs/heads/main/docs/mcp/architecture/demo.gif)

## Features

- **Fully Containerized**: AI Toolkit and MCP server run in Docker containers
- **NVIDIA GPU Support**: Full CUDA 12.1 support for GPU-accelerated training
- **Persistent Storage**: All datasets, configs, and outputs are persisted via volume mounts
- **MCP Integration**: AI assistants can create training configs and manage training jobs via MCP tools
- **HTTP MCP API**: HTTP MCP API for easy integration with remote agents
- **Dataset Management**: Upload and manage training datasets via MCP
- **Smart Configuration**: Automatic test prompt generation with trigger word integration
- **Training Monitoring**: Real-time training status and progress tracking
- **Model Download**: Download trained models directly through MCP
- **Web UI**: Access AI Toolkit's web interface at http://localhost:8675

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with Docker GPU support (`nvidia-docker2`)
- CUDA-compatible GPU (CUDA 12.1)
- At least 30GB free disk space for models and datasets
- 24GB+ VRAM recommended for FLUX LoRA training (lower VRAM possible with `low_vram` mode)

## Quick Start

1. **Build the Docker images**:
   ```bash
   docker-compose build
   ```

2. **Start the services**:
   ```bash
   # Start AI Toolkit and standard MCP server
   docker-compose up -d
   
   # Optional: Start HTTP API server for easier integration
   docker-compose up -d mcp-http-server
   ```

   To use existing ComfyUI model folders from Windows/WSL, set these (or rely on defaults):
   ```bash
   export COMFYUI_REPO_G="/mnt/g/AI/REPO"
   export COMFYUI_REPO_F="/mnt/f/AI/REPO"
   export COMFYUI_MODELS_C="/mnt/c/Users/LIZ/Desktop/ComfyUI_windows_portable_nvidia_cu128/ComfyUI_windows_portable/ComfyUI/models"
   docker-compose up -d --build
   ```

3. **Access the services**:
   - AI Toolkit Web UI: http://localhost:8675
   - MCP HTTP API: http://localhost:8190 (if started)
   - MCP stdio: Available in the `mcp-ai-toolkit-trainer` container

## Directory Structure

```
model-trainer-mcp/
├── docker-compose.yml      # Container orchestration
├── Dockerfile             # Container image definition
├── mcp_server.py          # MCP server implementation
├── mcp_http_server.py     # HTTP API wrapper
├── requirements.txt       # Python dependencies
└── example_training.py    # Example usage script
```

Note: All files are at the root level for GitHub Gist compatibility. When deployed:
- Datasets are stored in Docker volumes at `/ai-toolkit/datasets/`
- Configs are saved to `/ai-toolkit/configs/`
- Outputs are written to `/ai-toolkit/outputs/`
- Logs are stored in `/ai-toolkit/logs/`

## MCP Tools Available

### Training Configuration

#### `create-training-config`
Create a new LoRA training configuration with customizable parameters.

**Parameters:**
- `name` (required): Name for the training job
- `model_name` (required): Base model identifier or local model path
  - HuggingFace ID: `ostris/Flex.1-alpha`
  - Local filename (auto-resolved from mounted model roots): `flux1-dev-fp8.safetensors`
  - Local absolute path in container: `/comfy-models/g_repo/checkpoint/flux1-dev-fp8.safetensors`
  - Note: Use publicly accessible models. `black-forest-labs/FLUX.1-dev` requires authentication
- `dataset_path` (required): Path to the dataset folder (e.g., "/ai-toolkit/datasets/my-dataset")
- `resolution`: Training resolution in pixels (default: 512)
- `batch_size`: Training batch size (default: 1)
- `learning_rate`: Learning rate (default: 0.0002)
- `steps`: Number of training steps (default: 1000)
- `rank`: LoRA rank - higher for more complex concepts (default: 16)
- `alpha`: LoRA alpha - typically same as rank (default: 16)
- `use_wandb`: Enable Weights & Biases logging (default: false)
- `low_vram`: Enable low VRAM mode for GPUs with <24GB (default: true)
- `trigger_word`: Unique trigger word for activating the LoRA
- `test_prompts`: Array of test prompts for validation (recommended: 4 prompts)
  - Include 3 similar prompts and 1 unique/creative prompt
  - All prompts must include the trigger word
  - If not provided, default prompts will be auto-generated
- `disable_sampling`: Disable sample image generation during training (default: false)
  - Useful for faster training when you don't need intermediate samples
  - Significantly reduces training time by skipping image generation

#### `validate-training-config`
Run preflight validation on a training config before starting a job.

Validation covers:
- Required fields
- Flux/Flex architecture and model-path compatibility
- Resource sanity checks (`resolution`, `batch_size`, `steps`, `learning_rate`, `rank`, `alpha`)
- Optional dataset diagnostics (existence, corrupt files, caption coverage, duplicates)

**Input modes (choose one):**
- `config_name`: Validate an already saved config
- `config`: Validate a raw config object or JSON/YAML string
- Draft parameters: `model_name` + `dataset_path` (+ optional train params)

**Parameters:**
- `config_name`: Saved config name to validate
- `config`: Raw config object or JSON/YAML string
- `model_name`: Draft model reference
- `dataset_path`: Draft dataset path
- `resolution`: Draft resolution (default: 512)
- `batch_size`: Draft batch size (default: 1)
- `learning_rate`: Draft learning rate (default: 0.0002)
- `steps`: Draft step count (default: 1000)
- `rank`: Draft LoRA rank (default: 16)
- `alpha`: Draft LoRA alpha (default: 16)
- `check_dataset`: Include dataset diagnostics (default: true)
- `max_dataset_files`: Max dataset files scanned (default: 2000)

**Returns:**
- `valid`: boolean
- `errors`: blocking issues
- `warnings`: non-blocking issues
- `suggested_fixes`: actionable recommendations
- `normalized_config`: normalized view of key training fields

**Examples:**

1. Validate a saved config:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "validate-training-config",
    "arguments": {
      "config_name": "my-flux-lora",
      "check_dataset": true
    }
  }'
```

2. Validate draft parameters before saving:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "validate-training-config",
    "arguments": {
      "model_name": "ostris/Flex.1-alpha",
      "dataset_path": "/ai-toolkit/datasets/my-dataset",
      "resolution": 1024,
      "steps": 2500,
      "rank": 32,
      "alpha": 32
    }
  }'
```

3. Validate a raw config payload:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "validate-training-config",
    "arguments": {
      "config": {
        "config": {
          "name": "raw-check",
          "process": [{
            "model": {"name_or_path": "ostris/Flex.1-alpha", "arch": "flex1", "low_vram": true},
            "datasets": [{"folder_path": "/ai-toolkit/datasets/my-dataset", "resolution": [1024]}],
            "train": {"batch_size": 1, "steps": 1500, "lr": 0.0002},
            "network": {"linear": 16, "linear_alpha": 16}
          }]
        }
      }
    }
  }'
```

#### `recommend-training-plan`
Generate an intelligent training plan (balanced + alternatives) from goal, dataset, VRAM target, and time budget.

**Parameters:**
- `goal`: Free-text objective (style/character/object/etc.)
- `base_model`: Base model reference (or `model_name` alias)
- `dataset_path`: Optional dataset path for diagnostics
- `config_name`: Optional saved config to infer model/dataset context
- `time_budget_minutes`: Optional max run budget
- `target_vram_gb`: Optional GPU memory target
- `include_dataset_scan`: Include dataset diagnostics (default: true)
- `max_dataset_files`: Max files scanned for dataset diagnostics (default: 2000)

**Returns:**
- `plan.recommended_plan`: Primary recommendation
- `plan.alternative_plans`: Fast/balanced/quality variants
- `plan.confidence`, `plan.rationale`, and warnings

**Examples:**

1. Plan from a clear goal and hardware budget:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "recommend-training-plan",
    "arguments": {
      "goal": "Train a painterly style LoRA for fantasy scenes",
      "base_model": "ostris/Flex.1-alpha",
      "dataset_path": "/ai-toolkit/datasets/fantasy-style",
      "target_vram_gb": 16,
      "time_budget_minutes": 180
    }
  }'
```

2. Plan by reusing an existing config context:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "recommend-training-plan",
    "arguments": {
      "config_name": "my-flux-lora",
      "goal": "Improve character consistency",
      "time_budget_minutes": 120
    }
  }'
```

3. Minimal call with defaults:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "recommend-training-plan",
    "arguments": {
      "goal": "General concept LoRA"
    }
  }'
```

#### `run-training-with-guardrails`
Run training with built-in guardrails:
1) preflight validation
2) start job
3) monitor for critical failure signals
4) auto-stop when configured guardrails trigger

Default guardrails:
- NaN/Inf loss detection
- repeated OOM detection
- stalled progress detection
- throughput collapse detection

**Key parameters:**
- `config_name` (required): Saved config to run
- `block_on_validation_fail`: Prevent launch on preflight errors (default: true)
- `monitor_seconds`: Post-launch monitor duration (default: 120)
- `poll_interval_seconds`: Poll interval during monitor window (default: 10)
- `oom_threshold`: OOM mentions threshold before stop (default: 3)
- `throughput_floor_iter_per_sec`: Low-throughput floor (default: 0.02)

**Returns:**
- `started`, `status`, `job_id`
- `validation` report
- `guardrails` config
- `triggered_guardrails` and `stop_result`
- `final_status`

**Examples:**

1. Safe default run:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "run-training-with-guardrails",
    "arguments": {
      "config_name": "my-flux-lora"
    }
  }'
```

2. Longer monitor window with stricter OOM stop:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "run-training-with-guardrails",
    "arguments": {
      "config_name": "my-flux-lora",
      "monitor_seconds": 300,
      "poll_interval_seconds": 10,
      "oom_threshold": 2
    }
  }'
```

3. Start even with validation warnings/errors (advanced):
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "run-training-with-guardrails",
    "arguments": {
      "config_name": "experimental-config",
      "block_on_validation_fail": false,
      "stop_on_throughput_collapse": true,
      "throughput_floor_iter_per_sec": 0.03
    }
  }'
```

#### `watch-training-timeseries`
Monitor a job for a bounded window and return structured time-series metrics.

The tool captures:
- per-poll step/progress/speed/loss/error/health
- optional system telemetry (GPU/CPU/RAM)
- optional alert objects
- aggregated summary (`step delta`, speed/loss trend, status/health counts)

**Parameters:**
- `job_id` (required): Training job ID
- `duration_seconds`: Watch duration (default: 120)
- `poll_interval_seconds`: Poll interval (default: 10)
- `log_lines`: Log lines analyzed each poll (default: 400)
- `include_system`: Include system telemetry (default: true)
- `include_alerts`: Include alerts in each point (default: true)
- `include_dataset`: Include one-time dataset diagnostics context (default: false)
- `include_nlp`: Include NLP summaries (default: false)
- `stop_on_terminal_status`: Stop early on terminal status (default: true)

**Examples:**

1. Basic watch:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "watch-training-timeseries",
    "arguments": {
      "job_id": "job_12345",
      "duration_seconds": 120,
      "poll_interval_seconds": 10
    }
  }'
```

2. High-detail watch with excerpts:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "watch-training-timeseries",
    "arguments": {
      "job_id": "job_12345",
      "duration_seconds": 300,
      "poll_interval_seconds": 15,
      "include_raw_log_excerpt": true,
      "excerpt_lines": 120
    }
  }'
```

3. Watch with dataset/NLP context:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "watch-training-timeseries",
    "arguments": {
      "job_id": "job_12345",
      "include_dataset": true,
      "include_nlp": true,
      "max_dataset_files": 1500
    }
  }'
```

#### `compare-training-runs`
Compare multiple jobs and rank them using weighted scores:
- convergence
- speed
- stability
- dataset quality (optional)

Each ranked item includes `delta_to_winner` with total and per-component score differences versus the top run for explainability, plus confidence bands (`negligible`, `moderate`, `large`) and `band_source` provenance (threshold method and values).

Sample `delta_to_winner` payload:

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

**Parameters:**
- `job_ids`: Optional explicit list of job IDs
- `limit`: If `job_ids` omitted, number of recent jobs to compare (default: 5)
- `log_lines`: Log lines per run (default: 800)
- `include_dataset`: Include dataset diagnostics in score (default: true)
- `include_nlp`: Include NLP diagnostics in run reports (default: false)
- `only_terminal_runs`: Restrict to completed/failed/stopped jobs (default: false)

**Examples:**

1. Compare explicit jobs:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "compare-training-runs",
    "arguments": {
      "job_ids": ["job_12345", "job_45678", "job_78901"],
      "include_dataset": true
    }
  }'
```

2. Compare recent terminal jobs:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "compare-training-runs",
    "arguments": {
      "limit": 8,
      "only_terminal_runs": true
    }
  }'
```

3. Fast compare without dataset scans:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "compare-training-runs",
    "arguments": {
      "limit": 5,
      "include_dataset": false,
      "log_lines": 400
    }
  }'
```

#### `evaluate-lora`
Evaluate one training run with rubric scoring and grade output.

Default rubric components:
- convergence
- stability
- speed
- dataset quality
- caption quality (when NLP is included)

**Parameters:**
- `job_id` (required): Training job ID
- `log_lines`: Log lines analyzed (default: 1000)
- `include_dataset`: Include dataset diagnostics in scoring (default: true)
- `include_nlp`: Include NLP/caption diagnostics in scoring (default: true)
- `pass_threshold`: Pass threshold for overall score (default: 70)
- `rubric_weights`: Optional custom weights

**Examples:**

1. Standard evaluation:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "evaluate-lora",
    "arguments": {
      "job_id": "job_12345"
    }
  }'
```

2. Faster evaluation without dataset scan:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "evaluate-lora",
    "arguments": {
      "job_id": "job_12345",
      "include_dataset": false,
      "include_nlp": false,
      "log_lines": 400
    }
  }'
```

3. Custom rubric and threshold:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "evaluate-lora",
    "arguments": {
      "job_id": "job_12345",
      "pass_threshold": 75,
      "rubric_weights": {
        "convergence": 0.4,
        "stability": 0.35,
        "speed": 0.15,
        "dataset": 0.1
      }
    }
  }'
```

#### `suggest-next-experiment`
Recommend a single highest-impact next change based on run diagnostics and evaluation.

The tool:
- evaluates candidate runs
- selects the best/primary candidate
- proposes one concrete next experiment with expected impact and risk

**Parameters:**
- `job_id`: Primary job ID
- `job_ids`: Optional candidate run IDs
- `include_dataset`: Include dataset signals (default: true)
- `include_nlp`: Include NLP/caption signals (default: true)
- `pass_threshold`: Evaluation pass threshold (default: 70)

**Examples:**

1. Suggest next step from one run:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "suggest-next-experiment",
    "arguments": {
      "job_id": "job_12345"
    }
  }'
```

2. Suggest from multiple candidate runs:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "suggest-next-experiment",
    "arguments": {
      "job_ids": ["job_12345", "job_45678", "job_78901"],
      "include_dataset": true,
      "include_nlp": true
    }
  }'
```

3. Quick recommendation without NLP/dataset overhead:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "suggest-next-experiment",
    "arguments": {
      "job_ids": ["job_12345", "job_45678"],
      "include_dataset": false,
      "include_nlp": false,
      "log_lines": 500
    }
  }'
```

#### `auto-improve-dataset`
Inspect a dataset for missing/empty captions, duplicate images, and corrupt files, with optional safe apply mode.

**Parameters:**
- `dataset_path`: Dataset directory path
- `config_name`: Optional config name used to infer dataset path
- `max_dataset_files`: Max files inspected (default: 5000)
- `apply`: Apply changes (default: false)
- `trigger_word`: Optional prefix for generated captions
- `fill_missing_captions`: Create missing `.txt` files (default: true)
- `fill_empty_captions`: Replace empty captions (default: true)
- `quarantine_duplicates`: Move duplicate files to `_duplicates` (default: false)
- `quarantine_corrupt`: Move corrupt files to `_quarantine` (default: true)

**Examples:**

1. Analysis-only preflight:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "auto-improve-dataset",
    "arguments": {
      "dataset_path": "/ai-toolkit/datasets/my-dataset",
      "apply": false
    }
  }'
```

2. Apply caption fixes only:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "auto-improve-dataset",
    "arguments": {
      "dataset_path": "/ai-toolkit/datasets/my-dataset",
      "apply": true,
      "trigger_word": "mytok",
      "fill_missing_captions": true,
      "fill_empty_captions": true,
      "quarantine_duplicates": false,
      "quarantine_corrupt": false
    }
  }'
```

3. Full cleanup including quarantine:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "auto-improve-dataset",
    "arguments": {
      "dataset_path": "/ai-toolkit/datasets/my-dataset",
      "apply": true,
      "quarantine_duplicates": true,
      "quarantine_corrupt": true
    }
  }'
```

#### `export-observability-report`
Generate reproducible observability artifacts as JSON and Markdown for one job.

**Parameters:**
- `job_id` (required): Training job ID
- `config_name`: Optional config context
- `dataset_path`: Optional dataset path override
- `output_dir`: Optional output directory (default: `/ai-toolkit/outputs/reports`)
- `filename_prefix`: Prefix for output filenames (default: `observability`)
- `lines`: Log lines analyzed (default: 800)
- `include_dataset`: Include dataset diagnostics (default: true)
- `include_nlp`: Include NLP diagnostics (default: true)
- `include_evaluation`: Include rubric output (default: true)
- `include_raw_log_excerpt`: Include raw log excerpt (default: false)

**Examples:**

1. Standard report export:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "export-observability-report",
    "arguments": {
      "job_id": "job_12345"
    }
  }'
```

2. Lightweight report:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "export-observability-report",
    "arguments": {
      "job_id": "job_12345",
      "include_dataset": false,
      "include_nlp": false,
      "include_evaluation": false,
      "lines": 400
    }
  }'
```

3. Custom output location/prefix:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "export-observability-report",
    "arguments": {
      "job_id": "job_12345",
      "output_dir": "/ai-toolkit/outputs/reports/team-a",
      "filename_prefix": "nightly"
    }
  }'
```

#### `alert-routing`
Route alerts to file or webhook sinks from explicit alerts or from a job-derived observability snapshot.

**Parameters:**
- `job_id`: Optional job ID used to derive alerts when `alerts` is omitted
- `alerts`: Optional explicit alerts array
- `sink`: `file` or `webhook` (default: `file`)
- `destination`: File path or webhook URL destination
- `min_severity`: `info`, `warning`, or `critical` (default: `warning`)
- `lines`: Log lines analyzed for job-derived alerts (default: 500)
- `include_context`: Include run context metadata (default: true)
- `state_path`: Optional state file path for dedupe/rate-limit memory
- `dedupe_window_seconds`: Suppress duplicate alert fingerprints inside this window (default: 900)
- `rate_limit_window_seconds`: Rate-limit window size in seconds (default: 900)
- `max_alerts_per_window`: Max routed alerts per window, `0` disables rate-limit (default: 50)
- `webhook_retry_attempts`: Webhook delivery attempts including first attempt (default: 3)
- `webhook_backoff_ms`: Initial webhook retry backoff in milliseconds (default: 500)
- `webhook_backoff_multiplier`: Exponential backoff multiplier (default: 2.0)
- `webhook_timeout_seconds`: Per-attempt webhook timeout in seconds (default: 15)
- `webhook_retry_status_codes`: Retryable webhook HTTP statuses (default: `[408,409,425,429,500,502,503,504]`)

**Examples:**

1. Route derived job alerts to default file sink:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "alert-routing",
    "arguments": {
      "job_id": "job_12345",
      "sink": "file"
    }
  }'
```

2. Route only critical alerts to a custom file:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "alert-routing",
    "arguments": {
      "job_id": "job_12345",
      "sink": "file",
      "destination": "/ai-toolkit/outputs/alerts/critical.jsonl",
      "min_severity": "critical",
      "dedupe_window_seconds": 1800,
      "rate_limit_window_seconds": 600,
      "max_alerts_per_window": 20
    }
  }'
```

3. Push explicit alerts to webhook:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "alert-routing",
    "arguments": {
      "sink": "webhook",
      "destination": "https://example.com/hooks/training-alerts",
      "alerts": [
        {
          "severity": "critical",
          "category": "training",
          "message": "NaN detected in loss stream"
        }
      ],
      "webhook_retry_attempts": 4,
      "webhook_backoff_ms": 750,
      "webhook_backoff_multiplier": 2.0,
      "webhook_timeout_seconds": 20,
      "webhook_retry_status_codes": [408, 429, 500, 502, 503, 504]
    }
  }'
```

#### `list-configs`
List all available training configurations.

#### `get-config`
Retrieve a specific training configuration by name.

#### `get-training-info`
Get training information including trigger word and test prompts for a configuration.

**Parameters:**
- `name` (required): Configuration name

### Dataset Management

#### `upload-dataset`
Upload images with captions to create a new training dataset.

**Parameters:**
- `dataset_name` (required): Name for the dataset
- `images` (required): Array of images with:
  - `filename`: Image filename
  - `content`: Base64-encoded image content
  - `caption`: Caption/description for the image

#### `list-datasets`
List available datasets with image counts.

#### `list-comfyui-models`
List local model files available from configured ComfyUI model roots.

**Parameters:**
- `query`: Optional filename/path substring filter
- `limit`: Maximum results to return (default: 50, max: 500)

### Training Management

#### `start-training`
Start a training job using a saved configuration.

**Parameters:**
- `config_name` (required): Name of the configuration to use

#### `get-training-status`
Get the current status of a training job.

**Parameters:**
- `job_id` (required): Training job ID

#### `stop-training`
Stop a running training job.

**Parameters:**
- `job_id` (required): Training job ID to stop

#### `list-training-jobs`
List all training jobs and their statuses.

### Model Export

#### `export-model`
Export a trained model in the specified format.

**Parameters:**
- `job_id` (required): Training job ID
- `format`: Export format ("safetensors" or "ckpt", default: "safetensors")

#### `list-exported-models`
List all trained models available in the outputs directory.

#### `download-model`
Download a trained LoRA model as base64-encoded content.

**Parameters:**
- `model_path` (required): Path to the model file (relative to outputs directory)
- `include_metadata`: Include training metadata if available (default: true)

**Returns:**
- Base64-encoded model content
- Model metadata (if available)
- Training configuration (if available)

### System Information

#### `get-system-stats`
Get AI Toolkit system statistics including GPU usage.

#### `get-training-logs`
Retrieve logs for a specific training job.

**Parameters:**
- `job_id` (required): Training job ID
- `lines`: Number of log lines to retrieve (default: 100)

#### `get-training-observability`
Generate a deep log-intelligence report for a training job with one MCP call.

The report combines:
- Speed checks (iterations/sec trend and throughput status)
- Quality checks (loss trend, convergence score, NaN/Inf detection)
- Error classification (OOM, traceback, IO/network/numerical categories)
- Dataset diagnostics (caption coverage, corrupt images, duplicates, resolution stats)
- NLP summaries (top terms from logs/captions, caption diversity heuristics)
- Rubric evaluation (weighted score + grade + pass/fail)
- Actionable intelligence (bottlenecks, recommendations, decision gates)
- Optional baseline ranking against other runs

**Parameters:**
- `job_id` (required): Training job ID
- `lines`: Number of log lines to analyze (default: 500)
- `config_name`: Optional config name used for dataset-path resolution
- `dataset_path`: Optional explicit dataset path override
- `include_dataset`: Include dataset diagnostics (default: true)
- `include_nlp`: Include NLP summaries (default: true)
- `include_raw_log_excerpt`: Include a raw log excerpt in output (default: false)
- `excerpt_lines`: Tail lines for raw excerpt (default: 120)
- `max_dataset_files`: Max images scanned during dataset checks (default: 5000)
- `include_evaluation`: Include weighted evaluation output (default: true)
- `include_next_experiment`: Include next-experiment recommendation (default: true)
- `pass_threshold`: Pass threshold for rubric score (default: 70)
- `include_baseline_comparison`: Compare against other runs and return rank/percentile (default: false)
- `baseline_job_ids`: Optional explicit baseline job IDs
- `baseline_limit`: Max recent baseline runs when baseline_job_ids omitted (default: 5)

**Behavior Notes:**
- Throughput and readiness gates in `intelligence` use adaptive thresholds based on inferred model family (`flux_flex`, `sdxl`, `sd3`, `sd_like`) and run class (`smoke`, `standard`, `highres`, `long`).
- Speed scoring itself uses adaptive per-family throughput curves (instead of fixed global speed bands), so slower model families/resolutions are scored against appropriate expectations.
- Baseline comparison now reports `error: "target_run_unavailable"` when the requested `job_id` cannot be collected, while still returning any collected baseline IDs and skipped-baseline details.
- `alerts[*].confidence` is normalized to `[0.1, 0.99]`, and `intelligence.bottlenecks[*].confidence` reuses the strongest matching alert-category confidence (with severity-based fallback when category alerts are unavailable). Each bottleneck also includes `confidence_source` (`alert_category_max` or `severity_default`).

Sample intelligence bottleneck payload:

```json
{
  "code": "THROUGHPUT_LOW",
  "severity": "warning",
  "message": "Observed throughput is below the adaptive healthy range for this run profile.",
  "confidence": 0.76,
  "confidence_source": "alert_category_max"
}
```

**Examples:**

1. Default deep observability + intelligence:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "get-training-observability",
    "arguments": {
      "job_id": "your-job-id",
      "lines": 1200
    }
  }'
```

2. Add baseline comparison against recent runs:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "get-training-observability",
    "arguments": {
      "job_id": "your-job-id",
      "include_baseline_comparison": true,
      "baseline_limit": 5
    }
  }'
```

3. Compare against explicit baseline jobs with stricter pass threshold:
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "get-training-observability",
    "arguments": {
      "job_id": "your-job-id",
      "pass_threshold": 80,
      "baseline_job_ids": ["job-a", "job-b", "job-c"],
      "include_baseline_comparison": true
    }
  }'
```

## Observability Benchmarking

Use the built-in benchmark harness to measure detector latency and fixture precision.

### Run benchmark on golden fixtures

```bash
python scripts/benchmark_observability.py \
  --fixtures tests/fixtures/observability_cases.json \
  --iterations 200
```

The command writes:
- machine-readable JSON report under `outputs/benchmarks/observability/`
- human-readable Markdown summary under `outputs/benchmarks/observability/`

### Generate synthetic logs for regression scenarios

```bash
python scripts/generate_synthetic_training_log.py \
  --scenario oom \
  --steps 40 \
  --output outputs/benchmarks/synthetic/oom.log
```

### End-to-end MCP path smoke checks

Run stdio + HTTP MCP smoke checks locally without requiring a live AI Toolkit backend:

```bash
python scripts/smoke_mcp_paths.py --mode both
```

This validates:
- stdio tool registration and execution via `handle_call_tool`
- HTTP tool routing via `/mcp/tool` and `/mcp/tools`
- alias mapping (for example `route-alerts` and `export-report`)
- alert dedupe/rate-limit behavior

### Make targets

Project-level automation targets:

```bash
# Full local CI smoke chain
make smoke-ci

# Individual targets
make test
make benchmark-observability
make synth-log
make smoke-mcp
```

## HTTP API Access

The MCP server can optionally be accessed via HTTP for easier integration.

### Starting the HTTP Server

```bash
docker-compose up -d mcp-http-server
```

### HTTP Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `POST /mcp/tool` - Execute any MCP tool
- `GET /mcp/tools` - List available tools
- `GET /datasets` - List datasets
- `GET /configs` - List training configurations

### HTTP Examples

#### Create a Training Configuration
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "create-training-config",
    "arguments": {
      "name": "my-flux-lora",
      "model_name": "ostris/Flex.1-alpha",
      "dataset_path": "/ai-toolkit/datasets/my-dataset",
      "steps": 2000,
      "rank": 32,
      "trigger_word": "my_style",
      "test_prompts": [
        "a photo of my_style artwork",
        "a detailed image of my_style",
        "a high quality picture of my_style",
        "my_style in a cyberpunk cityscape with neon lights"
      ]
    }
  }'
```

#### Get Training Information
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "get-training-info",
    "arguments": {
      "name": "my-flux-lora"
    }
  }'
```

#### Upload a Dataset
The upload-dataset tool automatically generates `.aitk_size.json` metadata files required by AI Toolkit for proper dataset recognition.

```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "upload-dataset",
    "arguments": {
      "dataset_name": "my-style-dataset",
      "images": [
        {
          "filename": "image1.jpg",
          "content": "base64_encoded_content_here",
          "caption": "a photo of my_style artwork"
        }
      ]
    }
  }'
```

#### Start Training
```bash
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "start-training",
    "arguments": {
      "config_name": "my-flux-lora"
    }
  }'
```

#### Download Trained Model
```bash
# List available models first
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "list-exported-models",
    "arguments": {}
  }'

# Download a specific model
curl -X POST http://localhost:8190/mcp/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "download-model",
    "arguments": {
      "model_path": "my-flux-lora/checkpoint-1000.safetensors"
    }
  }' > model_response.json

# Extract the base64 content and decode it
jq -r '.result | fromjson | .content' model_response.json | base64 -d > my_lora.safetensors
```

## Training Best Practices

For best results with LoRA training:

1. **Image Requirements:**
   - PNG or JPG format
   - Consistent aspect ratio (square images work best for FLUX)
   - High quality, diverse examples showing different angles/contexts
   - 10-50 images typically sufficient for style LoRAs
   - 20-100 images for character/object LoRAs

2. **Caption Format:**
   - Each image needs a corresponding `.txt` file with the same name
   - Include trigger words consistently in captions
   - Be descriptive but concise
   - Vary the descriptions while keeping the trigger word
   - Example: "a photo of my_style artwork, digital painting, vibrant colors"

3. **Dataset Structure:**
   ```
   datasets/my-dataset/
   ├── image1.jpg
   ├── image1.txt
   ├── image2.jpg
   ├── image2.txt
   └── ...
   ```

4. **Caption Best Practices:**
   - Place trigger word at the beginning: "my_style portrait of a woman"
   - Add variety: backgrounds, lighting, poses, contexts
   - Include style descriptors: "my_style, oil painting style, dramatic lighting"
   - Avoid repetitive captions - each should be unique

5. **Flux Model Caption Guidelines:**
   
   For Flux models specifically, ensure your dataset image captions cover these essential elements:
   
   - **Trigger Word**: Always include your unique trigger word (e.g., "my_style", "xyz_character")
   - **Subject**: Clearly describe what's in the photo (e.g., "a woman", "a landscape", "a robot")
   - **Angle/Perspective**: Specify the camera angle or viewpoint (e.g., "front view", "aerial shot", "close-up portrait", "three-quarter view")
   - **Environment/Setting**: Describe where the subject is located (e.g., "in a forest", "urban street", "studio background", "underwater")
   - **Lighting**: Include lighting conditions (e.g., "soft natural light", "dramatic rim lighting", "golden hour", "neon illumination")
   
   **Example Caption Structure:**
   ```
   "my_style portrait of a woman, three-quarter view, in a modern office, soft window lighting"
   "my_style robot, full body shot from below, in a cyberpunk cityscape, neon purple lighting"
   "my_style landscape, wide aerial view, mountain forest environment, sunset golden hour lighting"
   ```
   
   This comprehensive captioning helps Flux models better understand and reproduce your style across different contexts and conditions.


## Example Usage

See `example_training.py` for a complete example of using the MCP HTTP API to:
- Create training configurations
- Upload datasets
- Start and monitor training
- Download trained models

Run the example:
```bash
python example_training.py
```

## Supported Models

AI Toolkit supports training LoRAs for:

- **FLUX/Flex Models**: Latest state-of-the-art models
  - `ostris/Flex.1-alpha` (recommended - publicly accessible)
  - `ostris/Flux.1-dev` (if available)
  - Note: `black-forest-labs/FLUX.1-dev` requires authentication

- **Stable Diffusion Models**:
  - `runwayml/stable-diffusion-v1-5` (widely compatible)
  - `stabilityai/stable-diffusion-2-1`
  - `CompVis/stable-diffusion-v1-4`

- **SDXL Models** and other diffusion models supported by AI Toolkit

## Example Usage

See `example_training.py` for a complete example of using the MCP HTTP API to:
- Create training configurations
- Upload datasets
- Start and monitor training
- Download trained models

Run the example:
```bash
python example_training.py
```

## Supported Models

AI Toolkit supports training LoRAs for:
- **FLUX/Flex Models**: Latest state-of-the-art models
  - `ostris/Flex.1-alpha` (recommended - publicly accessible)
  - `ostris/Flux.1-dev` (if available)
  - Note: `black-forest-labs/FLUX.1-dev` requires authentication
- **Stable Diffusion Models**:
  - `runwayml/stable-diffusion-v1-5` (widely compatible)
  - `stabilityai/stable-diffusion-2-1`
  - `CompVis/stable-diffusion-v1-4`
- **SDXL Models** and other diffusion models supported by AI Toolkit

## Configuration

### Environment Variables
- `LOG_LEVEL`: Set logging level (default: INFO)
- `AI_TOOLKIT_SERVER_URL`: Override AI Toolkit server URL
- `MCP_HTTP_PORT`: HTTP API port (default: 8190)

### GPU Configuration
The setup uses all available NVIDIA GPUs by default. To limit GPU usage, modify `NVIDIA_VISIBLE_DEVICES` in docker-compose.yml.

## Training Tips

1. **Low VRAM Mode**: Enable `low_vram: true` in configurations for GPUs with <24GB VRAM
2. **Learning Rate**: 
   - Default is now 2e-4 (0.0002) for better training performance
   - Use 1e-4 (0.0001) for more conservative training
   - Use 5e-5 (0.00005) for fine-tuning existing styles
   - Increase to 3e-4 (0.0003) for stubborn concepts
3. **Steps**: 
   - **Formula**: 100 × number of images in dataset
   - Example: 20 images = 2000 steps, 30 images = 3000 steps
   - For single image: 100 steps is often sufficient
   - Monitor test images - stop early if overfitting occurs
   - Higher step counts may lead to overfitting on small datasets
4. **Rank Selection**:
   - 8-16: Simple styles or minor adjustments
   - 16-32: Standard character/style LoRAs
   - 32-64: Complex concepts or multiple subjects
5. **Test Prompts**: 
   - Always include 4 test prompts
   - 3 should be variations of your training data
   - 1 should test generalization (unique scenario)
6. **Trigger Words**:
   - Use unique, non-dictionary words
   - Avoid common words that might conflict
   - Examples: "xyz_style", "abc_character", "def_object"
7. **Sample Generation**:
   - Enable by default to monitor training progress
   - Disable with `disable_sampling: true` for faster training
   - Disabling saves ~20-30% training time on FLUX models

## Building the Container

```bash
# Build all services
docker-compose build
```

The build script automatically detects Linux systems and uses host network mode for better DNS resolution during the build process, which helps avoid network-related build failures.

## Troubleshooting

### Container won't start
- Check NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Ensure ports 8675/8190 are not already in use
- Verify Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Check for container name conflicts with other MCP services (all container names should be unique)

### Training fails to start
- Check dataset path exists and contains images with captions
- Verify GPU has sufficient VRAM (use `nvidia-smi` to check)
- Ensure all images have corresponding `.txt` caption files
- Check logs: `docker-compose logs mcp-ai-toolkit-trainer`
- If you see "Cannot read properties of undefined (reading 'process')", the job config format is incorrect
- If you see model not found errors, ensure you're using accessible models like `ostris/Flex.1-alpha`

### Dataset won't appear in Web UI
- Dataset must contain `.aitk_size.json` metadata file (automatically generated by upload-dataset tool)
- The metadata file contains image dimensions and file signatures in AI Toolkit's specific format
- For manual uploads, you'll need to generate this file with the correct format:
  ```json
  {
    "__version__": "0.1.2",
    "\\image1.jpg": [width, height, "filesize:hash"]
  }
  ```

### MCP connection issues
- Check logs: `docker-compose logs mcp-ai-toolkit-trainer`
- Ensure AI Toolkit is healthy: `docker-compose ps`
- Verify MCP server is running: `docker exec mcp-ai-toolkit-trainer ps aux | grep mcp`

### HTTP API not accessible
- Ensure mcp-http-server is running: `docker-compose ps mcp-http-server`
- Check logs: `docker-compose logs mcp-http-server`
- Verify port 8190 is accessible: `curl http://localhost:8190/health`

### Out of Memory errors
- Enable `low_vram: true` in your configuration
- Reduce batch size to 1
- Lower resolution to 512x512
- Reduce rank to 8 or 16

## Performance Notes

- **GPU Memory**: FLUX LoRA training requires ~20-24GB VRAM
- **Training Time**: 
  - SD 1.5: ~1.5-2 iter/sec on RTX 4090 (100 steps ≈ 1-2 minutes)
  - FLUX/Flex: ~0.3-0.5 iter/sec on RTX 4090 (100 steps ≈ 5-10 minutes)
  - 1000 steps: 30-60 minutes on RTX 4090
  - 3000 steps: 90-180 minutes on RTX 4090
- **Disk Space**: Each training run can use 5-10GB for checkpoints

## Remote Deployment

If deploying on a remote server:

1. **Update MCP configuration**: When containers run on a remote server, update your MCP client configuration to point to the correct host:
   ```json
   {
     "mcpServers": {
       "ai-toolkit": {
         "command": "docker",
         "args": ["exec", "-i", "mcp-ai-toolkit-trainer", "python", "/app/mcp_server.py"],
         "env": {
           "AI_TOOLKIT_SERVER_URL": "http://YOUR_REMOTE_HOST:8675"
         }
       }
     }
   }
   ```

2. **Access services remotely**:
   - AI Toolkit Web UI: `http://YOUR_REMOTE_HOST:8675`
   - MCP HTTP API: `http://YOUR_REMOTE_HOST:8190`

3. **Monitor training**: Use the Web UI or MCP tools to monitor training progress remotely

## Stopping the Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all datasets and outputs)
docker-compose down -v
```

### Known Issues
1. **Config Storage**: Configurations created via MCP are stored in the database and are NOT visible as files in the Web UI's config browser
2. **Model Access**: Some models like `black-forest-labs/FLUX.1-dev` require authentication; use publicly accessible alternatives like `ostris/Flex.1-alpha`
3. **Dataset Path**: When using `create-training-config`, the dataset_path should be just the dataset name (e.g., "my-dataset") not the full path

## Notes

### API Implementation Status
All MCP tools are fully functional with the AI Toolkit Web UI. The MCP server integrates with AI Toolkit's database and API endpoints.

**Fully functional tools:**
- ✅ Configuration management (create, list, get)
- ✅ Dataset upload and management (with automatic `.aitk_size.json` generation)
- ✅ Training job control (start, stop, status)
- ✅ Model listing and downloading
- ✅ Real-time training monitoring
- ✅ Training logs retrieval

**Important Notes:**
- Configurations created via MCP are stored in AI Toolkit's SQLite database
- These configs are NOT visible as files in the Web UI's config browser
- The Web UI and filesystem configs are separate systems
- Jobs created via MCP's `start-training` tool ARE visible in the Web UI
- The Web UI expects configs in this exact format:
  ```json
  {
    "job": "extension",
    "config": {
      "name": "job_name",
      "process": [{
        "type": "ui_trainer",
        // ... rest of config
      }]
    },
    "meta": {
      "name": "[name]",
      "version": "1.0"
    }
  }
  ```
- Database initialization is handled automatically during container build

## Technical Implementation Details

### Database Structure
- AI Toolkit uses Prisma ORM with SQLite database (`aitk_db.db`)
- Jobs are stored with `job_config` as a stringified JSON object
- The database is initialized during container build with `npm run update_db`

### Configuration Format
The MCP server generates configurations that match the Web UI's expected format:
- Job type must be `"extension"` (not `"train"` or other values)
- Trainer type must be `"ui_trainer"` (not `"sd_trainer"`)
- Config is wrapped in `{job, config, meta}` structure

### Dataset Metadata Generation
The `.aitk_size.json` file uses a specific format:
- File paths use backslash prefix: `"\\image.jpg"`
- Signature format: `"filesize:hash"` where hash is first 8 chars of MD5 as decimal
- Only first 1024 bytes are used for hash calculation

### Container Architecture
- AI Toolkit runs on port 8675 (Web UI)
- MCP server runs inside the ai-toolkit container
- Optional HTTP wrapper runs on port 8190
- All containers share volumes for datasets, configs, and outputs
