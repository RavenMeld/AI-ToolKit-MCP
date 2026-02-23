#!/usr/bin/env python3
import os
import json
import uuid
import asyncio
import aiohttp
import websockets
import base64
import math
import re
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging
from datetime import datetime, timezone
import yaml
import hashlib
from PIL import Image
from io import BytesIO

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Server configuration
AI_TOOLKIT_SERVER_URL = os.getenv('AI_TOOLKIT_SERVER_URL', 'http://localhost:8675')
DATASET_DIR = Path("/ai-toolkit/datasets")
OUTPUT_DIR = Path("/ai-toolkit/outputs")
CONFIG_DIR = Path("/ai-toolkit/configs")
LOG_DIR = Path("/workspace/logs")
REPORTS_DIR = OUTPUT_DIR / "reports"
ALERTS_DIR = OUTPUT_DIR / "alerts"
DEFAULT_COMFYUI_MODEL_ROOTS = [
    "/comfy-models/g_repo",
    "/comfy-models/f_repo",
    "/comfy-models/comfyui_models",
]
COMFYUI_MODEL_ROOTS = [
    Path(path_str)
    for path_str in os.getenv("COMFYUI_MODEL_ROOTS", ":".join(DEFAULT_COMFYUI_MODEL_ROOTS)).split(":")
    if path_str
]
LOCAL_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".bin", ".pt", ".pth", ".gguf"}

# MCP Server instance
server = Server("ai-toolkit-mcp")

# Available tools
MCP_TOOLS = [
    'create-training-config',
    'validate-training-config',
    'recommend-training-plan',
    'run-training-with-guardrails',
    'watch-training-timeseries',
    'compare-training-runs',
    'evaluate-lora',
    'suggest-next-experiment',
    'auto-improve-dataset',
    'export-observability-report',
    'alert-routing',
    'list-configs',
    'get-config',
    'get-training-info',
    'upload-dataset',
    'list-datasets',
    'list-comfyui-models',
    'start-training',
    'get-training-status',
    'stop-training',
    'list-training-jobs',
    'export-model',
    'list-exported-models',
    'download-model',
    'get-system-stats',
    'get-training-observability',
    'get-training-logs'
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEXT_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you", "are", "was", "were",
    "have", "has", "had", "but", "can", "not", "into", "out", "all", "any", "per", "sec",
    "step", "steps", "epoch", "train", "training", "loss", "info", "warn", "warning", "error",
    "debug", "model", "dataset", "batch", "size", "time", "eta", "cuda", "gpu", "cpu", "ram",
    "start", "stop", "running", "status", "job", "jobs", "log", "logs"
}

VALIDATION_LIMITS = {
    "resolution_min": 256,
    "resolution_max": 2048,
    "flux_resolution_recommended_min": 768,
    "batch_size_max_recommended": 8,
    "steps_min_recommended": 100,
    "steps_max_recommended": 20000,
    "learning_rate_min": 1e-6,
    "learning_rate_max": 1e-2,
    "learning_rate_high_warning": 1e-3,
    "rank_max_recommended": 256,
}
SEVERITY_ORDER = {"none": 0, "info": 1, "warning": 2, "critical": 3}
ALERT_ROUTING_DEFAULT_DEDUPE_WINDOW_SECONDS = 900
ALERT_ROUTING_DEFAULT_RATE_WINDOW_SECONDS = 900
ALERT_ROUTING_DEFAULT_MAX_ALERTS_PER_WINDOW = 50
ALERT_ROUTING_STATE_PATH = ALERTS_DIR / "alert-routing-state.json"
ALERT_ROUTING_DEFAULT_WEBHOOK_RETRY_ATTEMPTS = 3
ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MS = 500
ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MULTIPLIER = 2.0
ALERT_ROUTING_DEFAULT_WEBHOOK_TIMEOUT_SECONDS = 15
ALERT_ROUTING_DEFAULT_RETRY_STATUS_CODES = [408, 409, 425, 429, 500, 502, 503, 504]
DELTA_TO_WINNER_BAND_SOURCE_ID = "absolute_delta_thresholds_v1"
DELTA_TO_WINNER_BAND_UNITS = "score_points"
DELTA_TO_WINNER_NEGLIGIBLE_MAX = 2.0
DELTA_TO_WINNER_MODERATE_MAX = 8.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _delta_band_source_payload() -> Dict[str, Any]:
    """Return provenance metadata for compare-run delta confidence bands."""
    return {
        "id": DELTA_TO_WINNER_BAND_SOURCE_ID,
        "units": DELTA_TO_WINNER_BAND_UNITS,
        "thresholds": {
            "negligible_lte": DELTA_TO_WINNER_NEGLIGIBLE_MAX,
            "moderate_lte": DELTA_TO_WINNER_MODERATE_MAX,
            "large_gt": DELTA_TO_WINNER_MODERATE_MAX,
        },
    }


def get_available_model_roots() -> List[Path]:
    """Return configured local model roots that are accessible in this runtime."""
    return [root for root in COMFYUI_MODEL_ROOTS if root.exists() and root.is_dir()]


def list_comfyui_models(query: str = "", limit: int = 50) -> List[Dict[str, str]]:
    """
    List local model files from configured ComfyUI model roots.
    Search is filename/relative-path substring matching.
    """
    q = (query or "").strip().lower()
    max_items = max(1, min(limit, 500))
    results: List[Dict[str, str]] = []

    for root in get_available_model_roots():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in LOCAL_MODEL_EXTENSIONS:
                continue

            rel_path = str(path.relative_to(root))
            haystack = f"{path.name} {rel_path}".lower()
            if q and q not in haystack:
                continue

            results.append({
                "name": path.name,
                "path": str(path),
                "relative_path": rel_path,
                "root": str(root),
            })
            if len(results) >= max_items:
                return results

    return results


def resolve_model_reference(model_name: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a model reference to a local file path when possible.
    Returns (resolved_reference, source_root_or_hint).
    """
    requested = (model_name or "").strip()
    if not requested:
        return requested, None

    direct = Path(requested).expanduser()
    if direct.exists() and direct.is_file():
        return str(direct), "direct-path"

    normalized = requested.replace("\\", "/").lstrip("./")
    available_roots = get_available_model_roots()
    for root in available_roots:
        candidate = root / normalized
        if candidate.exists() and candidate.is_file():
            return str(candidate), str(root)

    suffix = Path(requested).suffix.lower()
    # Preserve HuggingFace repo IDs such as "org/model-name"
    if "/" in requested and suffix not in LOCAL_MODEL_EXTENSIONS:
        return requested, None

    target_names = {Path(requested).name.lower()}
    if not suffix:
        target_names.update({f"{Path(requested).name.lower()}{ext}" for ext in LOCAL_MODEL_EXTENSIONS})

    for root in available_roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in LOCAL_MODEL_EXTENSIONS:
                continue
            if path.name.lower() in target_names:
                return str(path), str(root)

    return requested, None


def is_diffusers_style_model_dir(path: Path) -> bool:
    """Best-effort check for a local diffusers-style model directory."""
    if not path.exists() or not path.is_dir():
        return False
    if (path / "model_index.json").exists():
        return True
    if (path / "transformer" / "config.json").exists():
        return True
    return False

class AIToolkitClient:
    """Client for interacting with AI Toolkit API"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.client_id = str(uuid.uuid4())
        self.ws = None
        
    async def connect_websocket(self):
        """Connect to AI Toolkit websocket if available"""
        try:
            ws_url = f"{self.server_url.replace('http', 'ws')}/ws?clientId={self.client_id}"
            self.ws = await websockets.connect(ws_url)
            logger.info(f"Connected to AI Toolkit websocket: {ws_url}")
        except Exception as e:
            logger.warning(f"Could not connect to websocket: {e}")
            
    async def disconnect_websocket(self):
        """Disconnect from AI Toolkit websocket"""
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def _fetch_jobs(self, session: aiohttp.ClientSession) -> List[dict]:
        """Fetch all jobs from AI Toolkit"""
        async with session.get(f"{self.server_url}/api/jobs") as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            if isinstance(data, dict):
                return data.get("jobs", []) or []
            return []

    async def _get_job_by_id(self, session: aiohttp.ClientSession, job_id: str) -> Optional[dict]:
        """Fetch a single job by ID using the query-param API"""
        async with session.get(f"{self.server_url}/api/jobs", params={"id": job_id}) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if isinstance(data, dict) and data.get("id") == job_id:
                return data
            return None

    async def _find_job_by_name(self, session: aiohttp.ClientSession, job_name: str) -> Optional[dict]:
        """Find a job by name from the jobs list"""
        jobs = await self._fetch_jobs(session)
        for job in jobs:
            if job.get("name") == job_name:
                return job
        return None

    @staticmethod
    def _extract_total_steps(job_data: dict) -> int:
        """Extract total training steps from the stored job config payload"""
        job_config = job_data.get("job_config")
        if not job_config:
            return 0

        try:
            parsed_config = json.loads(job_config) if isinstance(job_config, str) else job_config
            process = parsed_config.get("config", {}).get("process", [])
            if process and isinstance(process, list):
                steps = process[0].get("train", {}).get("steps", 0)
                return int(steps or 0)
        except Exception:
            return 0

        return 0
            
    async def start_training(self, config_path: str) -> dict:
        """Start a training job with the given configuration"""
        # First, load the config to get the job name
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        job_name = config.get("config", {}).get("name", "unnamed_job")
        
        # Create or update the job in the database
        async with aiohttp.ClientSession() as session:
            existing_job = await self._find_job_by_name(session, job_name)

            if existing_job:
                job_id = existing_job.get("id")
                if not job_id:
                    return {"success": False, "error": "Found existing job without an ID"}

                update_data = {
                    "id": job_id,
                    "name": job_name,
                    "gpu_ids": existing_job.get("gpu_ids", "0"),
                    "job_config": config
                }
                async with session.post(f"{self.server_url}/api/jobs", json=update_data) as update_resp:
                    if update_resp.status != 200:
                        error_text = await update_resp.text()
                        return {"success": False, "error": f"Failed to update existing job: {error_text}"}
            else:
                create_data = {
                    "name": job_name,
                    "gpu_ids": "0",  # Default to GPU 0, could be made configurable
                    "job_config": config  # Pass the config object directly, AI Toolkit will stringify it
                }
                async with session.post(f"{self.server_url}/api/jobs", json=create_data) as create_resp:
                    if create_resp.status == 409:
                        # Name race: fetch the now-existing job and continue.
                        existing_job = await self._find_job_by_name(session, job_name)
                        if not existing_job or not existing_job.get("id"):
                            error_text = await create_resp.text()
                            return {"success": False, "error": f"Failed to resolve existing job after 409: {error_text}"}
                        job_id = existing_job["id"]
                    elif create_resp.status == 200:
                        job_data = await create_resp.json()
                        job_id = job_data.get("id", job_name)
                    else:
                        error_text = await create_resp.text()
                        return {"success": False, "error": f"Failed to create job: {error_text}"}
            
            # Now start the job (AI Toolkit uses GET for starting jobs)
            async with session.get(f"{self.server_url}/api/jobs/{job_id}/start") as start_resp:
                if start_resp.status == 200:
                    # Ensure queue worker is running after startup/restart.
                    try:
                        async with session.get(f"{self.server_url}/api/queue/0/start") as queue_resp:
                            if queue_resp.status != 200:
                                logger.warning("Queue start returned status %s", queue_resp.status)
                    except Exception as queue_err:
                        logger.warning("Failed to auto-start queue 0: %s", queue_err)
                    return {"success": True, "job_id": job_id}
                else:
                    error_text = await start_resp.text()
                    return {"success": False, "error": f"Failed to start job: {error_text}"}
                
    async def get_training_status(self, job_id: str) -> dict:
        """Get the status of a training job"""
        async with aiohttp.ClientSession() as session:
            job_data = await self._get_job_by_id(session, job_id)
            if not job_data:
                return {"found": False}

            total_steps = self._extract_total_steps(job_data)
            current_step = int(job_data.get("step") or 0)
            progress = int((current_step / total_steps) * 100) if total_steps > 0 else 0

            return {
                "found": True,
                "status": job_data.get("status", "Unknown"),
                "progress": progress,
                "current_step": current_step,
                "total_steps": total_steps,
                "info": job_data.get("info", ""),
                "speed_string": job_data.get("speed_string", "")
            }

    async def get_training_job(self, job_id: str) -> dict:
        """Fetch raw job payload for deeper analysis use-cases."""
        async with aiohttp.ClientSession() as session:
            job_data = await self._get_job_by_id(session, job_id)
            if not job_data:
                return {"found": False}
            return {"found": True, "job": job_data}
                
    async def stop_training(self, job_id: str) -> dict:
        """Stop a running training job"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.server_url}/api/jobs/{job_id}/stop") as resp:
                if resp.status == 200:
                    return {"success": True}
                error_text = await resp.text()
                return {"success": False, "error": f"Failed to stop job: {error_text}"}
                
    async def get_system_stats(self) -> dict:
        """Get AI Toolkit system statistics"""
        async with aiohttp.ClientSession() as session:
            gpu_data: dict = {}
            cpu_data: dict = {}
            jobs_data: List[dict] = []

            try:
                async with session.get(f"{self.server_url}/api/gpu") as resp:
                    if resp.status == 200:
                        gpu_data = await resp.json()
            except Exception:
                gpu_data = {}

            try:
                async with session.get(f"{self.server_url}/api/cpu") as resp:
                    if resp.status == 200:
                        cpu_data = await resp.json()
            except Exception:
                cpu_data = {}

            try:
                jobs_data = await self._fetch_jobs(session)
            except Exception:
                jobs_data = []

            first_gpu = {}
            if isinstance(gpu_data, dict):
                gpus = gpu_data.get("gpus", []) or []
                if gpus:
                    first_gpu = gpus[0]

            mem_info = first_gpu.get("memory", {}) if isinstance(first_gpu, dict) else {}
            util_info = first_gpu.get("utilization", {}) if isinstance(first_gpu, dict) else {}
            total_mem_mb = float(cpu_data.get("totalMemory", 0) or 0)
            avail_mem_mb = float(cpu_data.get("availableMemory", 0) or 0)

            active_jobs = sum(1 for j in jobs_data if (j.get("status") or "").lower() in {"running", "queued"})

            return {
                # Backward-compatible keys expected by existing formatter
                "gpu_name": first_gpu.get("name", "Unknown"),
                "gpu_memory_used": int(mem_info.get("used", 0) or 0),
                "gpu_memory_total": int(mem_info.get("total", 0) or 0),
                "gpu_utilization": int(util_info.get("gpu", 0) or 0),
                "cpu_percent": float(cpu_data.get("currentLoad", 0) or 0),
                "ram_used": round(max(total_mem_mb - avail_mem_mb, 0) / 1024, 2),
                "ram_total": round(total_mem_mb / 1024, 2),
                "active_jobs": active_jobs,
                # Additional raw payloads
                "gpu": gpu_data,
                "cpu": cpu_data,
                "jobs_count": len(jobs_data)
            }

    async def list_training_jobs(self) -> dict:
        """List all training jobs from AI Toolkit"""
        async with aiohttp.ClientSession() as session:
            jobs = await self._fetch_jobs(session)
            out_jobs = []
            for job in jobs:
                total_steps = self._extract_total_steps(job)
                current_step = int(job.get("step") or 0)
                progress = int((current_step / total_steps) * 100) if total_steps > 0 else 0
                out_jobs.append({
                    "id": job.get("id"),
                    "name": job.get("name"),
                    "status": job.get("status"),
                    "step": current_step,
                    "total_steps": total_steps,
                    "progress": progress,
                    "gpu_ids": job.get("gpu_ids"),
                    "info": job.get("info", ""),
                    "speed_string": job.get("speed_string", "")
                })
            return {"success": True, "jobs": out_jobs}

    async def get_training_logs(self, job_id: str, lines: int = 100) -> dict:
        """Fetch job log text from AI Toolkit"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.server_url}/api/jobs/{job_id}/log") as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {"success": False, "error": f"Failed to fetch logs: {error_text}"}

                data = await resp.json()
                log_text = data.get("log", "") if isinstance(data, dict) else ""
                if lines and lines > 0:
                    split_lines = log_text.splitlines()
                    log_text = "\n".join(split_lines[-lines:])
                return {"success": True, "log": log_text}

    async def export_model(self, job_id: str, export_format: str = "safetensors") -> dict:
        """
        Export model metadata for a completed job.
        AI Toolkit writes model files directly; this tool resolves the latest matching file.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.server_url}/api/jobs/{job_id}/files") as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {"success": False, "error": f"Failed to list job files: {error_text}"}

                data = await resp.json()
                files = data.get("files", []) if isinstance(data, dict) else []
                if not files:
                    return {"success": False, "error": "No exported model files found for this job"}

                requested_ext = f".{export_format.lower().lstrip('.')}"
                matching_files = [f for f in files if str(f.get("path", "")).lower().endswith(requested_ext)]
                if not matching_files:
                    matching_files = files

                selected = matching_files[-1]
                selected_path = selected.get("path", "")

                # Normalize path to be relative to OUTPUT_DIR when possible.
                try:
                    relative_path = str(Path(selected_path).resolve().relative_to(OUTPUT_DIR.resolve()))
                except Exception:
                    relative_path = selected_path

                return {
                    "success": True,
                    "job_id": job_id,
                    "format": export_format,
                    "model_path": relative_path,
                    "size": selected.get("size", 0),
                    "available_files": files
                }

# Configuration management functions
def create_lora_config(
    name: str,
    model_name: str,
    dataset_path: str,
    resolution: int = 512,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    steps: int = 1000,
    rank: int = 16,
    alpha: int = 16,
    use_wandb: bool = False,
    low_vram: bool = True,
    trigger_word: Optional[str] = None,
    test_prompts: Optional[List[str]] = None,
    disable_sampling: bool = False
) -> dict:
    """Create a LoRA training configuration"""
    
    # Validate required parameters
    if not name:
        raise ValueError("Configuration name is required")
    if not model_name:
        raise ValueError("Model name is required")
    if not dataset_path:
        raise ValueError("Dataset path is required")
    
    # Determine if using FLUX/Flex or SD model
    is_flux = "flux" in model_name.lower() or "flex" in model_name.lower()
    
    # Match the Web UI format exactly
    config = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "ui_trainer",
                "training_folder": "/ai-toolkit/outputs",
                "sqlite_db_path": "./aitk_db.db",
                "device": "cuda",
                "trigger_word": trigger_word or "",
                "performance_log_every": 10,
                "network": {
                    "type": "lora",
                    "linear": rank,
                    "linear_alpha": alpha,
                    "conv": rank // 2,
                    "conv_alpha": alpha // 2,
                    "lokr_full_rank": True,
                    "lokr_factor": -1,
                    "network_kwargs": {
                        "ignore_if_contains": []
                    }
                },
                "save": {
                    "dtype": "bf16",
                    "save_every": max(250, steps // 10),
                    "max_step_saves_to_keep": 4,
                    "save_format": "diffusers",
                    "push_to_hub": False
                },
                "datasets": [{
                    "folder_path": dataset_path,
                    "control_path": None,
                    "mask_path": None,
                    "mask_min_value": 0.1,
                    "default_caption": "",
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": False,
                    "is_reg": False,
                    "network_weight": 1.0,
                    "resolution": [resolution] if isinstance(resolution, int) else resolution,
                    "controls": []
                }],
                "train": {
                    "batch_size": batch_size,
                    "bypass_guidance_embedding": True,
                    "steps": steps,
                    "gradient_accumulation": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch" if is_flux else "ddpm",
                    "optimizer": "adamw8bit",
                    "timestep_type": "sigmoid" if is_flux else "linear",
                    "content_or_style": "balanced",
                    "optimizer_params": {
                        "weight_decay": 0.0001
                    },
                    "unload_text_encoder": False,
                    "lr": learning_rate,
                    "ema_config": {
                        "use_ema": False,
                        "ema_decay": 0.99
                    },
                    "skip_first_sample": False,
                    "disable_sampling": disable_sampling,
                    "dtype": "bf16",
                    "diff_output_preservation": False,
                    "diff_output_preservation_multiplier": 1,
                    "diff_output_preservation_class": "person"
                },
                "model": {
                    "name_or_path": model_name,
                    "quantize": True,
                    "quantize_te": True,
                    "arch": "flex1" if is_flux else "sd15",
                    "low_vram": low_vram,
                    "model_kwargs": {}
                },
                "sample": {
                    "sampler": "flowmatch" if is_flux else "ddim",
                    "sample_every": max(250, steps // 10),
                    "width": 1024 if is_flux else resolution,
                    "height": 1024 if is_flux else resolution,
                    "prompts": test_prompts if test_prompts else [trigger_word] if trigger_word else [],
                    "neg": "",
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 4.0 if is_flux else 7.5,
                    "sample_steps": 25 if is_flux else 20,
                    "num_frames": 1,
                    "fps": 1
                }
            }]
        },
        "meta": {
            "name": "[name]",
            "version": "1.0"
        }
    }
    
    # Trigger word is already set in the config
    # Test prompts are already handled in the sample section
        
    # Add wandb configuration if enabled
    if use_wandb:
        config["config"]["process"][0]["wandb"] = {
            "enabled": True,
            "project": f"lora-training-{name}",
            "run_name": name
        }
        
    # Store training metadata separately for easy retrieval
    config["training_metadata"] = {
        "trigger_word": trigger_word,
        "test_prompts": config["config"]["process"][0]["sample"]["prompts"],
        "model_name": model_name,
        "dataset_path": dataset_path,
        "resolution": resolution,
        "steps": steps,
        "rank": rank,
        "alpha": alpha,
        "learning_rate": learning_rate
    }
        
    return config

def list_configs() -> List[str]:
    """List available training configurations"""
    configs = []
    if CONFIG_DIR.exists():
        for file in CONFIG_DIR.glob("*.yaml"):
            configs.append(file.stem)
    return sorted(configs)

def load_config(name: str) -> Optional[dict]:
    """Load a training configuration by name"""
    config_path = CONFIG_DIR / f"{name}.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def save_config(name: str, config: dict) -> str:
    """Save a training configuration"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(config_path)

def list_datasets() -> List[dict]:
    """List available datasets"""
    datasets = []
    if DATASET_DIR.exists():
        for dataset_dir in DATASET_DIR.iterdir():
            if dataset_dir.is_dir():
                # Count images in dataset
                image_count = len(list(dataset_dir.glob("*.jpg")) + 
                                list(dataset_dir.glob("*.png")) + 
                                list(dataset_dir.glob("*.jpeg")))
                datasets.append({
                    "name": dataset_dir.name,
                    "path": str(dataset_dir),
                    "image_count": image_count
                })
    return sorted(datasets, key=lambda x: x["name"])

def get_file_signature(file_data: bytes, file_size: int) -> str:
    """
    Generate a file signature compatible with AI Toolkit's format
    Format: filesize:hash (hash is first 8 chars of MD5 hex as decimal)
    """
    # Use first 1024 bytes for signature (same as AI Toolkit)
    data_for_hash = file_data[:1024]
    hash_hex = hashlib.md5(data_for_hash).hexdigest()[:8]
    hash_value = int(hash_hex, 16)
    return f"{file_size}:{hash_value}"

def create_aitk_metadata(dataset_path: Path) -> None:
    """
    Create .aitk_size.json metadata file for AI Toolkit compatibility
    """
    metadata = {
        "__version__": "0.1.2"
    }
    
    # Process all images in the dataset
    for img_file in dataset_path.glob("*.jpg"):
        with open(img_file, 'rb') as f:
            file_data = f.read()
        
        # Get image dimensions
        img = Image.open(BytesIO(file_data))
        width, height = img.size
        
        # Get file size and signature
        file_size = len(file_data)
        signature = get_file_signature(file_data, file_size)
        
        # Use relative path with backslash prefix (AI Toolkit format)
        file_key = f"\\{img_file.name}"
        metadata[file_key] = [width, height, signature]
    
    # Also process PNG files
    for img_file in dataset_path.glob("*.png"):
        with open(img_file, 'rb') as f:
            file_data = f.read()
        
        img = Image.open(BytesIO(file_data))
        width, height = img.size
        file_size = len(file_data)
        signature = get_file_signature(file_data, file_size)
        
        file_key = f"\\{img_file.name}"
        metadata[file_key] = [width, height, signature]
    
    # Save metadata file
    metadata_path = dataset_path / ".aitk_size.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def list_output_models() -> List[dict]:
    """List all trained models in the outputs directory"""
    models = []
    if OUTPUT_DIR.exists():
        # Look for safetensors and ckpt files recursively
        for model_file in OUTPUT_DIR.rglob("*.safetensors"):
            models.append({
                "name": model_file.name,
                "path": str(model_file.relative_to(OUTPUT_DIR)),
                "size": model_file.stat().st_size,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })
        for model_file in OUTPUT_DIR.rglob("*.ckpt"):
            models.append({
                "name": model_file.name,
                "path": str(model_file.relative_to(OUTPUT_DIR)),
                "size": model_file.stat().st_size,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })
    return sorted(models, key=lambda x: x["modified"], reverse=True)


def _add_validation_issue(
    bucket: List[Dict[str, Any]],
    *,
    code: str,
    message: str,
    field: Optional[str] = None,
    value: Any = None
) -> None:
    """Append a structured validation issue."""
    issue: Dict[str, Any] = {"code": code, "message": message}
    if field:
        issue["field"] = field
    if value is not None:
        issue["value"] = value
    bucket.append(issue)


def _append_suggestion(suggestions: List[str], text: str) -> None:
    """Append a suggestion only once, preserving order."""
    if text not in suggestions:
        suggestions.append(text)


def _to_int(value: Any) -> Optional[int]:
    """Best-effort integer coercion."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    """Best-effort float coercion."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_resolution_values(value: Any) -> List[int]:
    """Normalize resolution values from scalar/list forms into integer list."""
    if value is None:
        return []
    if isinstance(value, list):
        normalized: List[int] = []
        for item in value:
            item_int = _to_int(item)
            if item_int is not None:
                normalized.append(item_int)
        return normalized
    scalar = _to_int(value)
    return [scalar] if scalar is not None else []


def validate_training_config_payload(
    config_payload: Dict[str, Any],
    *,
    check_dataset: bool = True,
    max_dataset_files: int = 2000
) -> Dict[str, Any]:
    """
    Validate a training config payload and return structured errors/warnings.
    """
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    suggested_fixes: List[str] = []

    if not isinstance(config_payload, dict):
        return {
            "valid": False,
            "errors": [{"code": "INVALID_CONFIG_TYPE", "message": "Configuration payload must be an object/dict."}],
            "warnings": [],
            "suggested_fixes": ["Provide a dict/YAML/JSON config payload."],
            "normalized_config": {},
            "summary": {"error_count": 1, "warning_count": 0},
        }

    root_cfg = config_payload.get("config", config_payload)
    if not isinstance(root_cfg, dict):
        return {
            "valid": False,
            "errors": [{"code": "INVALID_CONFIG_ROOT", "message": "Missing/invalid top-level 'config' object."}],
            "warnings": [],
            "suggested_fixes": ["Ensure config has a top-level object with `config.process[0]`."],
            "normalized_config": {},
            "summary": {"error_count": 1, "warning_count": 0},
        }

    process = root_cfg.get("process")
    if not isinstance(process, list) or not process or not isinstance(process[0], dict):
        return {
            "valid": False,
            "errors": [{"code": "MISSING_PROCESS", "message": "Config must include `config.process[0]`."}],
            "warnings": [],
            "suggested_fixes": ["Create config via `create-training-config` then adjust fields as needed."],
            "normalized_config": {},
            "summary": {"error_count": 1, "warning_count": 0},
        }

    p0 = process[0]
    model_cfg = p0.get("model") if isinstance(p0.get("model"), dict) else {}
    train_cfg = p0.get("train") if isinstance(p0.get("train"), dict) else {}
    network_cfg = p0.get("network") if isinstance(p0.get("network"), dict) else {}
    sample_cfg = p0.get("sample") if isinstance(p0.get("sample"), dict) else {}

    datasets = p0.get("datasets")
    ds0 = datasets[0] if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict) else {}

    model_name = str(model_cfg.get("name_or_path") or "").strip()
    model_arch = str(model_cfg.get("arch") or "").strip()
    model_arch_l = model_arch.lower()
    model_name_l = model_name.lower()

    is_flux_model = "flux" in model_name_l or "flex" in model_name_l
    is_flux_arch = model_arch_l.startswith("flex") or model_arch_l.startswith("flux")

    dataset_path = str(ds0.get("folder_path") or ds0.get("dataset_path") or ds0.get("path") or "").strip()
    resolution_values = _normalize_resolution_values(ds0.get("resolution"))
    batch_size = _to_int(train_cfg.get("batch_size"))
    steps = _to_int(train_cfg.get("steps"))
    learning_rate = _to_float(train_cfg.get("lr"))
    rank = _to_int(network_cfg.get("linear"))
    alpha = _to_int(network_cfg.get("linear_alpha"))
    low_vram = bool(model_cfg.get("low_vram", False))

    trigger_word = str(p0.get("trigger_word") or "").strip()
    prompts = sample_cfg.get("prompts")
    if not isinstance(prompts, list):
        prompts = []

    if not model_name:
        _add_validation_issue(
            errors,
            code="MISSING_MODEL_NAME",
            message="Missing `config.process[0].model.name_or_path`.",
            field="config.process[0].model.name_or_path",
        )
        _append_suggestion(
            suggested_fixes,
            "Set `model.name_or_path` to a HuggingFace repo ID or mounted local model path."
        )

    if not dataset_path:
        _add_validation_issue(
            errors,
            code="MISSING_DATASET_PATH",
            message="Missing dataset folder path in `datasets[0]`.",
            field="config.process[0].datasets[0].folder_path",
        )
        _append_suggestion(
            suggested_fixes,
            "Set `datasets[0].folder_path` to an existing dataset directory."
        )

    if batch_size is None:
        _add_validation_issue(
            errors,
            code="MISSING_BATCH_SIZE",
            message="Missing `train.batch_size`.",
            field="config.process[0].train.batch_size",
        )
    elif batch_size <= 0:
        _add_validation_issue(
            errors,
            code="INVALID_BATCH_SIZE",
            message="Batch size must be > 0.",
            field="config.process[0].train.batch_size",
            value=batch_size,
        )
    elif batch_size > VALIDATION_LIMITS["batch_size_max_recommended"]:
        _add_validation_issue(
            warnings,
            code="BATCH_SIZE_HIGH",
            message=f"Batch size {batch_size} may cause OOM on many GPUs.",
            field="config.process[0].train.batch_size",
            value=batch_size,
        )
        _append_suggestion(suggested_fixes, "Reduce batch size or enable/keep `low_vram: true`.")

    if steps is None:
        _add_validation_issue(
            errors,
            code="MISSING_STEPS",
            message="Missing `train.steps`.",
            field="config.process[0].train.steps",
        )
    elif steps <= 0:
        _add_validation_issue(
            errors,
            code="INVALID_STEPS",
            message="Training steps must be > 0.",
            field="config.process[0].train.steps",
            value=steps,
        )
    elif steps < VALIDATION_LIMITS["steps_min_recommended"]:
        _add_validation_issue(
            warnings,
            code="STEPS_LOW",
            message=f"Steps={steps} is low; resulting LoRA may underfit.",
            field="config.process[0].train.steps",
            value=steps,
        )
        _append_suggestion(suggested_fixes, "Use at least 100 steps for meaningful fine-tuning.")
    elif steps > VALIDATION_LIMITS["steps_max_recommended"]:
        _add_validation_issue(
            warnings,
            code="STEPS_HIGH",
            message=f"Steps={steps} is high; monitor overfitting.",
            field="config.process[0].train.steps",
            value=steps,
        )
        _append_suggestion(suggested_fixes, "Enable periodic sampling and stop early if quality plateaus.")

    if learning_rate is None:
        _add_validation_issue(
            errors,
            code="MISSING_LEARNING_RATE",
            message="Missing `train.lr`.",
            field="config.process[0].train.lr",
        )
    elif learning_rate <= 0:
        _add_validation_issue(
            errors,
            code="INVALID_LEARNING_RATE",
            message="Learning rate must be > 0.",
            field="config.process[0].train.lr",
            value=learning_rate,
        )
    else:
        if learning_rate < VALIDATION_LIMITS["learning_rate_min"] or learning_rate > VALIDATION_LIMITS["learning_rate_max"]:
            _add_validation_issue(
                warnings,
                code="LEARNING_RATE_OUT_OF_RANGE",
                message=(
                    f"Learning rate {learning_rate:g} is outside recommended range "
                    f"[{VALIDATION_LIMITS['learning_rate_min']:g}, {VALIDATION_LIMITS['learning_rate_max']:g}]."
                ),
                field="config.process[0].train.lr",
                value=learning_rate,
            )
        elif learning_rate > VALIDATION_LIMITS["learning_rate_high_warning"]:
            _add_validation_issue(
                warnings,
                code="LEARNING_RATE_HIGH",
                message=f"Learning rate {learning_rate:g} may destabilize training.",
                field="config.process[0].train.lr",
                value=learning_rate,
            )
            _append_suggestion(suggested_fixes, "Consider lowering learning rate toward 1e-4 to 3e-4.")

    if rank is None:
        _add_validation_issue(
            errors,
            code="MISSING_RANK",
            message="Missing LoRA rank `network.linear`.",
            field="config.process[0].network.linear",
        )
    elif rank <= 0:
        _add_validation_issue(
            errors,
            code="INVALID_RANK",
            message="LoRA rank must be > 0.",
            field="config.process[0].network.linear",
            value=rank,
        )
    elif rank > VALIDATION_LIMITS["rank_max_recommended"]:
        _add_validation_issue(
            warnings,
            code="RANK_HIGH",
            message=f"LoRA rank {rank} is unusually high and can increase instability/VRAM usage.",
            field="config.process[0].network.linear",
            value=rank,
        )
        _append_suggestion(suggested_fixes, "Use rank in the 8-64 range for most LoRA jobs.")

    if alpha is not None:
        if alpha <= 0:
            _add_validation_issue(
                errors,
                code="INVALID_ALPHA",
                message="LoRA alpha must be > 0.",
                field="config.process[0].network.linear_alpha",
                value=alpha,
            )
        elif rank and alpha > rank * 2:
            _add_validation_issue(
                warnings,
                code="ALPHA_HIGH_VS_RANK",
                message=f"Alpha {alpha} is much larger than rank {rank}.",
                field="config.process[0].network.linear_alpha",
                value=alpha,
            )
            _append_suggestion(suggested_fixes, "Keep LoRA alpha close to rank for stable scaling.")

    if resolution_values:
        for idx, rv in enumerate(resolution_values):
            if rv <= 0:
                _add_validation_issue(
                    errors,
                    code="INVALID_RESOLUTION",
                    message="Resolution values must be > 0.",
                    field=f"config.process[0].datasets[0].resolution[{idx}]",
                    value=rv,
                )
            elif rv < VALIDATION_LIMITS["resolution_min"]:
                _add_validation_issue(
                    warnings,
                    code="RESOLUTION_LOW",
                    message=f"Resolution {rv}px is low and may reduce output fidelity.",
                    field=f"config.process[0].datasets[0].resolution[{idx}]",
                    value=rv,
                )
                _append_suggestion(suggested_fixes, "Use 512px+ for general LoRA quality unless intentionally tiny.")
            elif rv > VALIDATION_LIMITS["resolution_max"]:
                _add_validation_issue(
                    warnings,
                    code="RESOLUTION_HIGH",
                    message=f"Resolution {rv}px is high and may significantly increase VRAM/runtime cost.",
                    field=f"config.process[0].datasets[0].resolution[{idx}]",
                    value=rv,
                )
                _append_suggestion(suggested_fixes, "Lower resolution or reduce batch size if memory pressure appears.")

        if is_flux_model and max(resolution_values) < VALIDATION_LIMITS["flux_resolution_recommended_min"]:
            _add_validation_issue(
                warnings,
                code="FLUX_RESOLUTION_LOW",
                message="Flux/Flex training is usually better at higher resolutions (>=768).",
                field="config.process[0].datasets[0].resolution",
                value=resolution_values,
            )

    if is_flux_model and not is_flux_arch:
        _add_validation_issue(
            errors,
            code="MODEL_ARCH_MISMATCH",
            message=f"Model '{model_name}' appears Flux/Flex but arch is '{model_arch or 'unset'}'.",
            field="config.process[0].model.arch",
            value=model_arch,
        )
        _append_suggestion(suggested_fixes, "Set `model.arch` to `flex1` for Flux/Flex model training.")
    elif not is_flux_model and is_flux_arch:
        _add_validation_issue(
            warnings,
            code="MODEL_ARCH_UNEXPECTED",
            message=f"Arch '{model_arch}' looks Flux/Flex, but model name does not.",
            field="config.process[0].model.arch",
            value=model_arch,
        )

    resolved_model = model_name
    model_resolution_source = None
    if model_name:
        resolved_model, model_resolution_source = resolve_model_reference(model_name)
        resolved_path = Path(str(resolved_model)).expanduser()
        if is_flux_model and resolved_path.exists():
            if resolved_path.is_file():
                _add_validation_issue(
                    errors,
                    code="FLUX_MODEL_FILE_NOT_ALLOWED",
                    message=(
                        "Flux/Flex training cannot use a single checkpoint file path. "
                        "Use a diffusers model ID or local diffusers directory."
                    ),
                    field="config.process[0].model.name_or_path",
                    value=str(resolved_path),
                )
                _append_suggestion(
                    suggested_fixes,
                    "Switch model to `ostris/Flex.1-alpha` or a local directory containing model_index.json."
                )
            elif resolved_path.is_dir() and not is_diffusers_style_model_dir(resolved_path):
                _add_validation_issue(
                    errors,
                    code="FLUX_MODEL_DIR_INVALID",
                    message="Flux/Flex local model directory is missing diffusers config files.",
                    field="config.process[0].model.name_or_path",
                    value=str(resolved_path),
                )
                _append_suggestion(
                    suggested_fixes,
                    "Ensure local model directory contains model_index.json or transformer/config.json."
                )
        if not resolved_path.exists():
            looks_like_local_checkpoint = (
                Path(model_name).suffix.lower() in LOCAL_MODEL_EXTENSIONS and "/" not in model_name
            )
            if looks_like_local_checkpoint:
                _add_validation_issue(
                    warnings,
                    code="MODEL_PATH_UNRESOLVED",
                    message=f"Model reference '{model_name}' did not resolve to a local file in configured model roots.",
                    field="config.process[0].model.name_or_path",
                    value=model_name,
                )
                _append_suggestion(
                    suggested_fixes,
                    "Use `list-comfyui-models` to find an available local model path."
                )

    dataset_exists = False
    dataset_stats: Optional[Dict[str, Any]] = None
    if dataset_path and check_dataset:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            _add_validation_issue(
                errors,
                code="DATASET_PATH_NOT_FOUND",
                message=f"Dataset path does not exist: {dataset_path}",
                field="config.process[0].datasets[0].folder_path",
                value=dataset_path,
            )
            _append_suggestion(suggested_fixes, "Create/upload dataset first or correct the dataset path.")
        elif not dataset_dir.is_dir():
            _add_validation_issue(
                errors,
                code="DATASET_PATH_NOT_DIRECTORY",
                message=f"Dataset path is not a directory: {dataset_path}",
                field="config.process[0].datasets[0].folder_path",
                value=dataset_path,
            )
        else:
            dataset_exists = True
            dataset_stats, _ = analyze_dataset_folder(dataset_dir, max_files=max_dataset_files)
            total_images = int(dataset_stats.get("total_images", 0))
            if total_images <= 0:
                _add_validation_issue(
                    errors,
                    code="DATASET_EMPTY",
                    message="No image files found in dataset directory.",
                    field="config.process[0].datasets[0].folder_path",
                    value=dataset_path,
                )
                _append_suggestion(
                    suggested_fixes,
                    "Upload image files (.jpg/.jpeg/.png/.webp/.bmp) to the dataset directory."
                )
            if int(dataset_stats.get("corrupt_image_count", 0)) > 0:
                _add_validation_issue(
                    errors,
                    code="DATASET_CORRUPT_IMAGES",
                    message=f"Found {dataset_stats.get('corrupt_image_count')} corrupt image files.",
                    field="config.process[0].datasets[0].folder_path",
                )
                _append_suggestion(suggested_fixes, "Remove or replace corrupt images before training.")
            caption_coverage = float(dataset_stats.get("caption_coverage_pct", 0.0))
            if caption_coverage < 80:
                _add_validation_issue(
                    warnings,
                    code="DATASET_LOW_CAPTION_COVERAGE",
                    message=f"Caption coverage is low ({caption_coverage:.1f}%).",
                    field="config.process[0].datasets[0].folder_path",
                    value=caption_coverage,
                )
                _append_suggestion(suggested_fixes, "Add caption .txt files for more training images.")
            duplicate_estimate = int(dataset_stats.get("duplicate_image_estimate", 0))
            if duplicate_estimate > 0:
                _add_validation_issue(
                    warnings,
                    code="DATASET_DUPLICATES_DETECTED",
                    message=f"Estimated duplicate images: {duplicate_estimate}.",
                    field="config.process[0].datasets[0].folder_path",
                    value=duplicate_estimate,
                )
                _append_suggestion(suggested_fixes, "Deduplicate dataset to improve concept diversity.")

    if is_flux_model and not low_vram:
        _add_validation_issue(
            warnings,
            code="LOW_VRAM_DISABLED_FOR_FLUX",
            message="Flux/Flex with `low_vram=false` may exceed memory on many GPUs.",
            field="config.process[0].model.low_vram",
            value=low_vram,
        )

    if trigger_word and prompts:
        missing_trigger = [p for p in prompts if isinstance(p, str) and trigger_word.lower() not in p.lower()]
        if missing_trigger:
            _add_validation_issue(
                warnings,
                code="PROMPTS_MISSING_TRIGGER",
                message=f"{len(missing_trigger)} sample prompt(s) do not include trigger word '{trigger_word}'.",
                field="config.process[0].sample.prompts",
            )
            _append_suggestion(suggested_fixes, "Include trigger word in all sample prompts for better checkpoint validation.")

    normalized_config = {
        "job_name": root_cfg.get("name"),
        "model": {
            "name_or_path": model_name or None,
            "resolved_name_or_path": resolved_model if model_name else None,
            "resolution_source": model_resolution_source,
            "arch": model_arch or None,
            "is_flux_model": is_flux_model,
            "is_flux_arch": is_flux_arch,
            "low_vram": low_vram,
        },
        "dataset": {
            "path": dataset_path or None,
            "exists": dataset_exists,
            "stats": dataset_stats if dataset_stats else {"checked": bool(check_dataset and dataset_path)},
        },
        "train": {
            "resolution": resolution_values,
            "batch_size": batch_size,
            "steps": steps,
            "learning_rate": learning_rate,
            "rank": rank,
            "alpha": alpha,
        },
        "sample": {
            "prompt_count": len(prompts),
            "trigger_word": trigger_word or None,
        },
    }

    if not suggested_fixes and not errors and not warnings:
        _append_suggestion(suggested_fixes, "Configuration looks healthy. Proceed to start training.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggested_fixes": suggested_fixes,
        "normalized_config": normalized_config,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "is_flux_model": is_flux_model,
            "dataset_checked": bool(check_dataset),
        },
    }


def _infer_goal_profile(goal: str) -> str:
    """Infer a coarse training goal profile from free text."""
    text = (goal or "").lower()
    if any(token in text for token in ("style", "aesthetic", "look", "art style", "painting")):
        return "style"
    if any(token in text for token in ("character", "person", "face", "portrait", "avatar")):
        return "character"
    if any(token in text for token in ("object", "product", "item", "logo")):
        return "object"
    return "general"


def recommend_training_plan_payload(
    *,
    goal: str,
    base_model: str,
    dataset_analysis: Optional[Dict[str, Any]],
    target_vram_gb: Optional[float],
    time_budget_minutes: Optional[float]
) -> Dict[str, Any]:
    """Generate heuristic recommendations for training setup and hyperparameters."""
    model_text = (base_model or "").lower()
    is_flux_model = "flux" in model_text or "flex" in model_text
    goal_profile = _infer_goal_profile(goal)

    dataset_size = 0
    caption_coverage = None
    duplicate_estimate = 0
    if dataset_analysis and dataset_analysis.get("exists"):
        dataset_size = int(dataset_analysis.get("total_images", 0) or 0)
        caption_cov = dataset_analysis.get("caption_coverage_pct")
        caption_coverage = float(caption_cov) if caption_cov is not None else None
        duplicate_estimate = int(dataset_analysis.get("duplicate_image_estimate", 0) or 0)

    if dataset_size <= 0:
        baseline_steps = 1000
    elif dataset_size < 20:
        baseline_steps = 700
    elif dataset_size < 100:
        baseline_steps = 1400
    elif dataset_size < 500:
        baseline_steps = 2600
    elif dataset_size < 2000:
        baseline_steps = 4200
    else:
        baseline_steps = 6000

    rank_base = {
        "style": 32,
        "character": 24,
        "object": 16,
        "general": 16,
    }.get(goal_profile, 16)
    if dataset_size and dataset_size < 30:
        rank_base = max(8, rank_base // 2)

    if dataset_size and dataset_size < 50:
        lr_base = 1.5e-4
    else:
        lr_base = 2e-4

    vram = target_vram_gb if target_vram_gb is not None else 16.0
    if is_flux_model:
        if vram <= 10:
            resolution_base = 640
            batch_base = 1
            low_vram = True
        elif vram <= 16:
            resolution_base = 768
            batch_base = 1
            low_vram = True
        elif vram <= 24:
            resolution_base = 1024
            batch_base = 1
            low_vram = False
        else:
            resolution_base = 1024
            batch_base = 2
            low_vram = False
    else:
        if vram <= 8:
            resolution_base = 512
            batch_base = 1
            low_vram = True
        elif vram <= 12:
            resolution_base = 640
            batch_base = 1
            low_vram = True
        elif vram <= 24:
            resolution_base = 768
            batch_base = 1
            low_vram = False
        else:
            resolution_base = 1024
            batch_base = 2
            low_vram = False

    throughput_steps_per_min = (18.0 if is_flux_model else 45.0) * (512.0 / float(resolution_base)) ** 2 * max(batch_base, 1)
    throughput_steps_per_min = max(4.0, throughput_steps_per_min)

    profiles = [
        ("fast", 0.65, 0.9),
        ("balanced", 1.0, 1.0),
        ("quality", 1.45, 0.8),
    ]
    alternatives = []
    for profile_name, step_mult, lr_mult in profiles:
        steps = max(300, int(round(baseline_steps * step_mult)))
        lr = lr_base * lr_mult
        if time_budget_minutes and time_budget_minutes > 0:
            estimated_minutes = steps / throughput_steps_per_min
            if estimated_minutes > time_budget_minutes:
                adjusted_steps = int(max(250, math.floor(time_budget_minutes * throughput_steps_per_min)))
                steps = min(steps, adjusted_steps)
        est_minutes = round(steps / throughput_steps_per_min, 1)
        alternatives.append({
            "profile": profile_name,
            "model_name": base_model or ("ostris/Flex.1-alpha" if is_flux_model else "runwayml/stable-diffusion-v1-5"),
            "resolution": resolution_base,
            "batch_size": batch_base,
            "learning_rate": float(f"{lr:.8f}"),
            "steps": steps,
            "rank": rank_base,
            "alpha": rank_base,
            "low_vram": low_vram,
            "estimated_minutes": est_minutes,
        })

    recommended = next((p for p in alternatives if p["profile"] == "balanced"), alternatives[0])

    confidence = 0.55
    if dataset_size > 0:
        confidence += 0.15
    if target_vram_gb is not None:
        confidence += 0.1
    if base_model:
        confidence += 0.1
    if caption_coverage is not None and caption_coverage < 80:
        confidence -= 0.05
    confidence = round(max(0.1, min(0.95, confidence)), 2)

    warnings: List[str] = []
    if dataset_size == 0:
        warnings.append("Dataset statistics unavailable; recommendations use conservative defaults.")
    elif dataset_size < 20:
        warnings.append("Very small dataset; expect overfitting risk and unstable generalization.")
    if caption_coverage is not None and caption_coverage < 80:
        warnings.append(f"Caption coverage is low ({caption_coverage:.1f}%).")
    if duplicate_estimate > 0:
        warnings.append(f"Potential duplicate images detected ({duplicate_estimate}).")
    if time_budget_minutes and recommended["estimated_minutes"] > time_budget_minutes:
        warnings.append("Balanced profile was clipped to match the provided time budget.")

    rationale = [
        f"Goal profile inferred as '{goal_profile}'.",
        f"Model family inferred as {'Flux/Flex' if is_flux_model else 'SD-like'} based on base model reference.",
        f"Baseline steps derived from dataset size ({dataset_size} images).",
        f"Resource settings adapted to target VRAM ({vram:g} GB).",
    ]

    return {
        "goal_profile": goal_profile,
        "is_flux_model": is_flux_model,
        "confidence": confidence,
        "recommended_plan": recommended,
        "alternative_plans": alternatives,
        "assumptions": {
            "dataset_images": dataset_size,
            "caption_coverage_pct": caption_coverage,
            "target_vram_gb": target_vram_gb,
            "time_budget_minutes": time_budget_minutes,
            "throughput_steps_per_min": round(throughput_steps_per_min, 2),
        },
        "warnings": warnings,
        "rationale": rationale,
    }


def detect_guardrail_triggers(
    *,
    quality_analysis: Dict[str, Any],
    error_analysis: Dict[str, Any],
    speed_analysis: Dict[str, Any],
    current_step: int,
    last_progress_step: int,
    seconds_since_progress: float,
    consecutive_low_speed: int,
    guardrails: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int]:
    """Evaluate guardrail conditions and return trigger list + updated speed counter."""
    triggers: List[Dict[str, Any]] = []

    stop_on_nan = bool(guardrails.get("stop_on_nan", True))
    stop_on_repeated_oom = bool(guardrails.get("stop_on_repeated_oom", True))
    stop_on_stalled_progress = bool(guardrails.get("stop_on_stalled_progress", True))
    stop_on_throughput_collapse = bool(guardrails.get("stop_on_throughput_collapse", True))
    oom_threshold = int(guardrails.get("oom_threshold", 3))
    stall_window_seconds = float(guardrails.get("stall_window_seconds", 120))
    min_step_delta = int(guardrails.get("min_step_delta", 1))
    throughput_floor_iter_per_sec = float(guardrails.get("throughput_floor_iter_per_sec", 0.02))
    low_speed_consecutive_polls = int(guardrails.get("low_speed_consecutive_polls", 2))

    quality_flags = set(quality_analysis.get("flags", []))
    if stop_on_nan and ("loss_nan_detected" in quality_flags or "loss_inf_detected" in quality_flags):
        triggers.append({
            "code": "TRAINING_NAN_LOSS",
            "severity": "critical",
            "message": "Detected NaN/Inf in loss values.",
            "evidence": {"flags": sorted(quality_flags)},
        })

    oom_count = int((error_analysis.get("category_counts") or {}).get("oom", 0) or 0)
    if stop_on_repeated_oom and oom_count >= oom_threshold:
        triggers.append({
            "code": "TRAINING_REPEATED_OOM",
            "severity": "critical",
            "message": f"OOM mentions reached threshold ({oom_count} >= {oom_threshold}).",
            "evidence": {"oom_count": oom_count, "oom_threshold": oom_threshold},
        })

    speed_latest = (speed_analysis.get("iter_per_sec") or {}).get("latest")
    if stop_on_throughput_collapse and speed_latest is not None:
        try:
            speed_value = float(speed_latest)
        except (TypeError, ValueError):
            speed_value = None
        if speed_value is not None and speed_value <= throughput_floor_iter_per_sec:
            consecutive_low_speed += 1
        else:
            consecutive_low_speed = 0

        if consecutive_low_speed >= low_speed_consecutive_polls:
            triggers.append({
                "code": "TRAINING_THROUGHPUT_COLLAPSE",
                "severity": "critical",
                "message": (
                    f"Throughput stayed below floor for {consecutive_low_speed} polls "
                    f"({speed_value} <= {throughput_floor_iter_per_sec})."
                ),
                "evidence": {
                    "latest_iter_per_sec": speed_value,
                    "throughput_floor_iter_per_sec": throughput_floor_iter_per_sec,
                    "consecutive_low_speed_polls": consecutive_low_speed,
                },
            })
    else:
        consecutive_low_speed = 0

    if stop_on_stalled_progress:
        required_progress = last_progress_step + max(1, min_step_delta)
        if seconds_since_progress >= stall_window_seconds and current_step < required_progress:
            triggers.append({
                "code": "TRAINING_STALLED_PROGRESS",
                "severity": "critical",
                "message": (
                    f"No sufficient step progress for {seconds_since_progress:.1f}s "
                    f"(current={current_step}, required>={required_progress})."
                ),
                "evidence": {
                    "seconds_since_progress": round(seconds_since_progress, 2),
                    "stall_window_seconds": stall_window_seconds,
                    "current_step": current_step,
                    "required_step": required_progress,
                },
            })

    return triggers, consecutive_low_speed


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp numeric values to a bounded range."""
    return max(lower, min(upper, value))


def _max_alert_severity(alerts: List[Dict[str, Any]]) -> str:
    """Return max severity for a list of alert objects."""
    severities = {str(alert.get("severity", "")).lower() for alert in alerts}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    if "info" in severities:
        return "info"
    return "none"


def summarize_timeseries_points(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize point-in-time observations from watch-training-timeseries."""
    if not points:
        return {
            "point_count": 0,
            "step": {"start": None, "end": None, "delta": None},
            "iter_per_sec": {"mean": None, "max": None, "trend_pct": None},
            "loss": {"start": None, "end": None, "delta_pct": None},
            "gpu": {"util_mean_pct": None, "memory_peak_mb": None},
            "status_counts": {},
            "health_counts": {},
            "alerts": {"total_alerts": 0, "critical_points": 0, "warning_points": 0},
        }

    steps = [_to_int(p.get("current_step")) for p in points]
    steps_filtered = [s for s in steps if s is not None]
    start_step = steps_filtered[0] if steps_filtered else None
    end_step = steps_filtered[-1] if steps_filtered else None

    speeds = [_to_float(p.get("iter_per_sec")) for p in points]
    speeds_filtered = [s for s in speeds if s is not None]
    speed_mean = round(statistics.fmean(speeds_filtered), 6) if speeds_filtered else None
    speed_max = round(max(speeds_filtered), 6) if speeds_filtered else None
    speed_trend_pct = None
    if len(speeds_filtered) >= 2 and speeds_filtered[0] > 0:
        speed_trend_pct = round(((speeds_filtered[-1] - speeds_filtered[0]) / speeds_filtered[0]) * 100, 2)

    losses = [_to_float(p.get("loss_last")) for p in points]
    losses_filtered = [l for l in losses if l is not None]
    loss_start = losses_filtered[0] if losses_filtered else None
    loss_end = losses_filtered[-1] if losses_filtered else None
    loss_delta_pct = None
    if loss_start is not None and loss_end is not None and loss_start != 0:
        loss_delta_pct = round(((loss_end - loss_start) / abs(loss_start)) * 100, 2)

    gpu_utils = [_to_float(p.get("gpu_utilization_pct")) for p in points]
    gpu_utils_filtered = [g for g in gpu_utils if g is not None]
    gpu_mem = [_to_float(p.get("gpu_memory_used_mb")) for p in points]
    gpu_mem_filtered = [m for m in gpu_mem if m is not None]

    status_counts = Counter(str(p.get("status", "unknown")).lower() for p in points)
    health_counts = Counter(str(p.get("overall_health", "unknown")).lower() for p in points)
    total_alerts = sum(int(_to_int(p.get("alert_count")) or 0) for p in points)
    critical_points = sum(1 for p in points if str(p.get("alert_max_severity", "")).lower() == "critical")
    warning_points = sum(1 for p in points if str(p.get("alert_max_severity", "")).lower() == "warning")

    return {
        "point_count": len(points),
        "step": {
            "start": start_step,
            "end": end_step,
            "delta": (end_step - start_step) if start_step is not None and end_step is not None else None,
        },
        "iter_per_sec": {
            "mean": speed_mean,
            "max": speed_max,
            "trend_pct": speed_trend_pct,
        },
        "loss": {
            "start": round(loss_start, 6) if loss_start is not None else None,
            "end": round(loss_end, 6) if loss_end is not None else None,
            "delta_pct": loss_delta_pct,
        },
        "gpu": {
            "util_mean_pct": round(statistics.fmean(gpu_utils_filtered), 4) if gpu_utils_filtered else None,
            "memory_peak_mb": round(max(gpu_mem_filtered), 4) if gpu_mem_filtered else None,
        },
        "status_counts": dict(status_counts),
        "health_counts": dict(health_counts),
        "alerts": {
            "total_alerts": total_alerts,
            "critical_points": critical_points,
            "warning_points": warning_points,
        },
    }


def score_dataset_quality(dataset_analysis: Optional[Dict[str, Any]]) -> Optional[float]:
    """Compute a dataset quality score (0-100) from diagnostics."""
    if not dataset_analysis or not dataset_analysis.get("exists"):
        return None

    scanned = int(dataset_analysis.get("scanned_images", 0) or dataset_analysis.get("total_images", 0) or 0)
    if scanned <= 0:
        return 0.0

    caption_coverage = float(dataset_analysis.get("caption_coverage_pct", 0.0) or 0.0)
    duplicate_estimate = int(dataset_analysis.get("duplicate_image_estimate", 0) or 0)
    corrupt_count = int(dataset_analysis.get("corrupt_image_count", 0) or 0)
    low_resolution_count = int(dataset_analysis.get("low_resolution_count", 0) or 0)
    extreme_aspect_count = int(dataset_analysis.get("extreme_aspect_ratio_count", 0) or 0)

    duplicate_ratio_pct = (duplicate_estimate / scanned) * 100.0
    low_res_ratio_pct = (low_resolution_count / scanned) * 100.0
    extreme_aspect_ratio_pct = (extreme_aspect_count / scanned) * 100.0

    score = 100.0
    score -= max(0.0, 100.0 - caption_coverage) * 0.45
    score -= min(40.0, duplicate_ratio_pct * 0.9)
    score -= min(60.0, float(corrupt_count) * 12.0)
    score -= min(20.0, low_res_ratio_pct * 0.3)
    score -= min(15.0, extreme_aspect_ratio_pct * 0.25)

    return round(_clamp(score, 0.0, 100.0), 2)


def score_stability(
    quality_analysis: Dict[str, Any],
    error_analysis: Dict[str, Any],
    alerts: List[Dict[str, Any]]
) -> float:
    """Compute run stability score (0-100) from quality, errors, and alerts."""
    score = 100.0

    error_severity = str(error_analysis.get("severity", "none")).lower()
    if error_severity == "critical":
        score -= 45.0
    elif error_severity == "warning":
        score -= 20.0

    total_mentions = int(error_analysis.get("total_mentions", 0) or 0)
    score -= min(30.0, total_mentions * 2.0)

    quality_status = str(quality_analysis.get("status", "unknown")).lower()
    if quality_status == "critical":
        score -= 40.0
    elif quality_status == "warning":
        score -= 15.0

    quality_flags = set(quality_analysis.get("flags", []) or [])
    if "loss_nan_detected" in quality_flags or "loss_inf_detected" in quality_flags:
        score = min(score, 10.0)
    if "loss_spikes_detected" in quality_flags:
        score -= 10.0

    for alert in alerts:
        sev = str(alert.get("severity", "")).lower()
        if sev == "critical":
            score -= 10.0
        elif sev == "warning":
            score -= 3.0

    return round(_clamp(score, 0.0, 100.0), 2)


def rank_run_comparisons(run_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank run reports using normalized component scoring."""
    if not run_reports:
        return {
            "ranking": [],
            "winner_job_id": None,
            "winner_score": None,
            "score_spread": None,
        }

    speed_values: List[float] = []
    for report in run_reports:
        speed_mean = _to_float((report.get("metrics") or {}).get("speed_mean_iter_per_sec"))
        if speed_mean is not None and speed_mean > 0:
            speed_values.append(speed_mean)

    speed_min = min(speed_values) if speed_values else None
    speed_max = max(speed_values) if speed_values else None

    scored_reports: List[Dict[str, Any]] = []
    for report in run_reports:
        metrics = report.get("metrics", {})
        convergence_score = _to_float(metrics.get("convergence_score"))
        if convergence_score is None:
            convergence_score = 50.0

        stability_score = _to_float(metrics.get("stability_score"))
        if stability_score is None:
            stability_score = 50.0

        speed_mean = _to_float(metrics.get("speed_mean_iter_per_sec"))
        if speed_mean is None or speed_mean <= 0:
            speed_score = 0.0
        elif speed_min is not None and speed_max is not None and speed_max > speed_min:
            speed_score = ((speed_mean - speed_min) / (speed_max - speed_min)) * 100.0
        else:
            speed_score = 80.0

        dataset_quality = _to_float(metrics.get("dataset_quality_score"))
        weight_convergence = 0.4
        weight_speed = 0.2
        weight_stability = 0.3
        weight_data = 0.1 if dataset_quality is not None else 0.0
        weight_total = weight_convergence + weight_speed + weight_stability + weight_data
        if weight_total <= 0:
            weight_total = 1.0

        weighted_sum = (
            (convergence_score * weight_convergence)
            + (speed_score * weight_speed)
            + (stability_score * weight_stability)
            + ((dataset_quality or 0.0) * weight_data)
        )
        total_score = round(weighted_sum / weight_total, 2)

        scored = dict(report)
        scored["scores"] = {
            "total": total_score,
            "components": {
                "convergence": round(convergence_score, 2),
                "speed": round(speed_score, 2),
                "stability": round(stability_score, 2),
                "dataset": round(dataset_quality, 2) if dataset_quality is not None else None,
            },
            "weights": {
                "convergence": weight_convergence,
                "speed": weight_speed,
                "stability": weight_stability,
                "dataset": weight_data,
            },
        }
        scored_reports.append(scored)

    scored_reports.sort(key=lambda item: item.get("scores", {}).get("total", 0), reverse=True)
    for idx, report in enumerate(scored_reports, start=1):
        report["rank"] = idx

    top_score = scored_reports[0]["scores"]["total"]
    winner_components = (scored_reports[0].get("scores") or {}).get("components", {})

    delta_band_source = _delta_band_source_payload()

    def _delta_band(delta: Optional[float]) -> Optional[str]:
        value = _to_float(delta)
        if value is None:
            return None
        magnitude = abs(value)
        if magnitude <= DELTA_TO_WINNER_NEGLIGIBLE_MAX:
            return "negligible"
        if magnitude <= DELTA_TO_WINNER_MODERATE_MAX:
            return "moderate"
        return "large"

    for report in scored_reports:
        report_components = (report.get("scores") or {}).get("components", {})
        component_delta: Dict[str, Optional[float]] = {}
        component_bands: Dict[str, Optional[str]] = {}
        for key, winner_value in winner_components.items():
            report_value = report_components.get(key)
            wv = _to_float(winner_value)
            rv = _to_float(report_value)
            if wv is None or rv is None:
                component_delta[key] = None
                component_bands[key] = None
            else:
                delta = round(rv - wv, 2)
                component_delta[key] = delta
                component_bands[key] = _delta_band(delta)

        total_delta = round(report["scores"]["total"] - top_score, 2)
        report["delta_to_winner"] = {
            "total": total_delta,
            "total_band": _delta_band(total_delta),
            "components": component_delta,
            "component_bands": component_bands,
            "band_source": {
                "id": delta_band_source["id"],
                "units": delta_band_source["units"],
                "thresholds": dict(delta_band_source["thresholds"]),
            },
        }

    bottom_score = scored_reports[-1]["scores"]["total"]

    return {
        "ranking": scored_reports,
        "winner_job_id": scored_reports[0].get("job_id"),
        "winner_score": top_score,
        "score_spread": round(top_score - bottom_score, 2),
    }


def _speed_curve_thresholds(model_family: str, run_class: str) -> Dict[str, float]:
    """Return adaptive throughput thresholds used for single-run speed scoring."""
    base = {
        "excellent": 0.30,
        "good": 0.20,
        "fair": 0.12,
        "watch": 0.08,
        "poor": 0.04,
    }
    family_multiplier = {
        "flux_flex": 0.55,
        "sdxl": 0.70,
        "sd3": 0.70,
        "sd_like": 1.00,
        "unknown": 1.00,
    }.get(model_family, 1.00)
    class_multiplier = {
        "smoke": 0.80,
        "standard": 1.00,
        "highres": 0.72,
        "long": 1.08,
    }.get(run_class, 1.00)
    factor = family_multiplier * class_multiplier
    return {k: round(v * factor, 4) for k, v in base.items()}


def _score_speed_absolute(
    mean_iter_per_sec: Optional[float],
    *,
    model_family: str = "unknown",
    run_class: str = "standard",
) -> Optional[float]:
    """Score speed for a single run without cross-run normalization."""
    if mean_iter_per_sec is None:
        return None
    speed = float(mean_iter_per_sec)
    if speed <= 0:
        return 0.0
    curve = _speed_curve_thresholds(model_family, run_class)
    if speed >= curve["excellent"]:
        return 95.0
    if speed >= curve["good"]:
        return 82.0
    if speed >= curve["fair"]:
        return 68.0
    if speed >= curve["watch"]:
        return 52.0
    if speed >= curve["poor"]:
        return 35.0
    return 20.0


def _score_caption_quality(nlp_analysis: Optional[Dict[str, Any]]) -> Optional[float]:
    """Score caption/NLP quality from lightweight signals."""
    if not nlp_analysis or nlp_analysis.get("skipped"):
        return None

    diversity = _to_float(nlp_analysis.get("caption_token_diversity"))
    style_flags = set(nlp_analysis.get("style_flags", []) or [])
    if diversity is None:
        return None

    if diversity >= 0.45:
        score = 92.0
    elif diversity >= 0.30:
        score = 78.0
    elif diversity >= 0.20:
        score = 63.0
    else:
        score = 45.0

    if "caption_vocabulary_low_diversity" in style_flags:
        score -= 12.0
    if "captions_too_short" in style_flags:
        score -= 15.0

    return round(_clamp(score, 0.0, 100.0), 2)


def evaluate_run_report(
    run_report: Dict[str, Any],
    *,
    rubric_weights: Optional[Dict[str, float]] = None,
    pass_threshold: float = 70.0
) -> Dict[str, Any]:
    """Evaluate one run report with rubric scoring and grade."""
    metrics = run_report.get("metrics", {}) if isinstance(run_report, dict) else {}
    quality_analysis = run_report.get("quality_analysis", {}) if isinstance(run_report, dict) else {}
    nlp_analysis = run_report.get("nlp_analysis", {}) if isinstance(run_report, dict) else {}
    config_payload = run_report.get("config_payload") if isinstance(run_report.get("config_payload"), dict) else None
    hparams = _extract_hparams_from_config(config_payload)
    model_family = _infer_model_family_from_config(config_payload)
    run_class = _infer_run_class_from_hparams(hparams)

    convergence_score = _to_float(metrics.get("convergence_score"))
    if convergence_score is None:
        convergence_score = 50.0

    stability_score = _to_float(metrics.get("stability_score"))
    if stability_score is None:
        stability_score = 50.0

    dataset_quality_score = _to_float(metrics.get("dataset_quality_score"))
    speed_abs_score = _score_speed_absolute(
        _to_float(metrics.get("speed_mean_iter_per_sec")),
        model_family=model_family,
        run_class=run_class,
    )
    caption_quality_score = _score_caption_quality(nlp_analysis if isinstance(nlp_analysis, dict) else None)

    default_weights = {
        "convergence": 0.35,
        "stability": 0.30,
        "speed": 0.20,
        "dataset": 0.10,
        "caption_quality": 0.05,
    }
    weights = dict(default_weights)
    if rubric_weights:
        for key, val in rubric_weights.items():
            if key in weights:
                parsed = _to_float(val)
                if parsed is not None and parsed >= 0:
                    weights[key] = parsed

    components = {
        "convergence": round(convergence_score, 2),
        "stability": round(stability_score, 2),
        "speed": round(speed_abs_score, 2) if speed_abs_score is not None else None,
        "dataset": round(dataset_quality_score, 2) if dataset_quality_score is not None else None,
        "caption_quality": round(caption_quality_score, 2) if caption_quality_score is not None else None,
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for key, score in components.items():
        if score is None:
            continue
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        weighted_sum += score * weight
        weight_total += weight

    overall = round(weighted_sum / weight_total, 2) if weight_total > 0 else 0.0
    pass_threshold = float(_clamp(pass_threshold, 0.0, 100.0))
    passed = overall >= pass_threshold

    if overall >= 90:
        grade = "A"
    elif overall >= 80:
        grade = "B"
    elif overall >= 70:
        grade = "C"
    elif overall >= 60:
        grade = "D"
    else:
        grade = "F"

    diagnostics: List[str] = []
    if quality_analysis.get("status") == "critical":
        diagnostics.append("Loss is unstable (NaN/Inf or severe numerical issues).")
    if quality_analysis.get("trend") == "regressing":
        diagnostics.append("Loss trend is regressing.")
    if components["dataset"] is not None and components["dataset"] < 70:
        diagnostics.append("Dataset quality appears to be a limiting factor.")
    if components["speed"] is not None and components["speed"] < 50:
        diagnostics.append("Training throughput is low; runtime efficiency may be limiting iteration count.")
    if components["caption_quality"] is not None and components["caption_quality"] < 65:
        diagnostics.append("Caption quality/diversity is weak.")

    return {
        "overall_score": overall,
        "grade": grade,
        "passed": passed,
        "pass_threshold": pass_threshold,
        "components": components,
        "weights": weights,
        "diagnostics": diagnostics,
    }


def _infer_model_family_from_config(config_payload: Optional[Dict[str, Any]]) -> str:
    """Infer broad model family from config payload."""
    if not config_payload:
        return "unknown"
    cfg = config_payload.get("config", config_payload) if isinstance(config_payload, dict) else {}
    if not isinstance(cfg, dict):
        return "unknown"
    process = cfg.get("process", [])
    if not isinstance(process, list) or not process or not isinstance(process[0], dict):
        return "unknown"

    p0 = process[0]
    model_cfg = p0.get("model") if isinstance(p0.get("model"), dict) else {}
    model_name = str(model_cfg.get("name_or_path") or "").strip().lower()
    model_arch = str(model_cfg.get("arch") or "").strip().lower()
    combined = f"{model_name} {model_arch}".strip()
    if not combined:
        return "unknown"
    if "flux" in combined or "flex" in combined:
        return "flux_flex"
    if "sdxl" in combined:
        return "sdxl"
    if "sd3" in combined or "stable-diffusion-3" in combined:
        return "sd3"
    return "sd_like"


def _infer_run_class_from_hparams(hparams: Dict[str, Any]) -> str:
    """Infer run class from hyperparameters."""
    steps = _to_int(hparams.get("steps"))
    resolution = _to_int(hparams.get("resolution"))
    if steps is not None and steps <= 1200:
        return "smoke"
    if steps is not None and steps >= 4500:
        return "long"
    if resolution is not None and resolution >= 1024:
        return "highres"
    return "standard"


def _adaptive_intelligence_thresholds(model_family: str, run_class: str) -> Dict[str, float]:
    """Compute model/run-class-aware decision thresholds for intelligence gating."""
    thresholds = {
        "ready_overall_min": 75.0,
        "safe_export_overall_min": 75.0,
        "speed_bottleneck_score": 50.0,
        "dataset_repair_score": 70.0,
        "caption_repair_score": 65.0,
    }

    if model_family == "flux_flex":
        thresholds["speed_bottleneck_score"] -= 12.0
        thresholds["ready_overall_min"] -= 1.0
    elif model_family in {"sdxl", "sd3"}:
        thresholds["speed_bottleneck_score"] -= 8.0

    if run_class == "smoke":
        thresholds["ready_overall_min"] -= 8.0
        thresholds["safe_export_overall_min"] -= 8.0
    elif run_class == "highres":
        thresholds["speed_bottleneck_score"] -= 8.0
        thresholds["ready_overall_min"] -= 2.0
        thresholds["safe_export_overall_min"] -= 2.0
    elif run_class == "long":
        thresholds["ready_overall_min"] += 3.0
        thresholds["safe_export_overall_min"] += 5.0
        thresholds["dataset_repair_score"] += 3.0
        thresholds["caption_repair_score"] += 3.0

    thresholds["ready_overall_min"] = round(_clamp(thresholds["ready_overall_min"], 60.0, 90.0), 2)
    thresholds["safe_export_overall_min"] = round(
        _clamp(max(thresholds["safe_export_overall_min"], thresholds["ready_overall_min"]), 60.0, 95.0),
        2,
    )
    thresholds["speed_bottleneck_score"] = round(_clamp(thresholds["speed_bottleneck_score"], 20.0, 70.0), 2)
    thresholds["dataset_repair_score"] = round(_clamp(thresholds["dataset_repair_score"], 55.0, 85.0), 2)
    thresholds["caption_repair_score"] = round(_clamp(thresholds["caption_repair_score"], 50.0, 85.0), 2)
    return thresholds


def build_log_intelligence_profile(
    snapshot: Dict[str, Any],
    run_report: Optional[Dict[str, Any]],
    evaluation: Optional[Dict[str, Any]],
    *,
    config_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build actionable intelligence from a single observability snapshot."""
    run_report = run_report or {}
    quality_analysis = snapshot.get("quality_analysis", {}) if isinstance(snapshot.get("quality_analysis"), dict) else {}
    speed_analysis = snapshot.get("speed_analysis", {}) if isinstance(snapshot.get("speed_analysis"), dict) else {}
    error_analysis = snapshot.get("error_analysis", {}) if isinstance(snapshot.get("error_analysis"), dict) else {}
    dataset_analysis = snapshot.get("dataset_analysis", {}) if isinstance(snapshot.get("dataset_analysis"), dict) else {}
    nlp_analysis = snapshot.get("nlp_analysis", {}) if isinstance(snapshot.get("nlp_analysis"), dict) else {}
    alerts = snapshot.get("alerts", []) if isinstance(snapshot.get("alerts"), list) else []

    resolved_config_payload: Optional[Dict[str, Any]] = None
    if isinstance(config_payload, dict):
        resolved_config_payload = config_payload
    else:
        embedded_payload = run_report.get("config_payload") if isinstance(run_report, dict) else None
        if isinstance(embedded_payload, dict):
            resolved_config_payload = embedded_payload
    hparams = _extract_hparams_from_config(resolved_config_payload)
    model_family = _infer_model_family_from_config(resolved_config_payload)
    run_class = _infer_run_class_from_hparams(hparams)
    adaptive_thresholds = _adaptive_intelligence_thresholds(model_family, run_class)

    metrics = run_report.get("metrics", {}) if isinstance(run_report.get("metrics"), dict) else {}
    convergence_score = _to_float(metrics.get("convergence_score"))
    if convergence_score is None:
        convergence_score = _to_float(quality_analysis.get("convergence_score"))
    if convergence_score is None:
        convergence_score = 50.0

    stability_score = _to_float(metrics.get("stability_score"))
    if stability_score is None:
        stability_score = score_stability(quality_analysis, error_analysis, alerts)

    speed_mean = _to_float(metrics.get("speed_mean_iter_per_sec"))
    if speed_mean is None:
        speed_mean = _to_float((speed_analysis.get("iter_per_sec") or {}).get("mean"))
    speed_score = _score_speed_absolute(
        speed_mean,
        model_family=model_family,
        run_class=run_class,
    )
    if speed_score is None:
        speed_score = 0.0

    dataset_score = _to_float(metrics.get("dataset_quality_score"))
    if dataset_score is None:
        dataset_score = score_dataset_quality(dataset_analysis)

    caption_score = _score_caption_quality(nlp_analysis)

    component_scores = {
        "convergence": round(convergence_score, 2),
        "stability": round(stability_score, 2),
        "speed": round(speed_score, 2),
        "dataset": round(dataset_score, 2) if dataset_score is not None else None,
        "caption_quality": round(caption_score, 2) if caption_score is not None else None,
    }
    component_weights = {
        "convergence": 0.30,
        "stability": 0.30,
        "speed": 0.20,
        "dataset": 0.12,
        "caption_quality": 0.08,
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for key, score in component_scores.items():
        if score is None:
            continue
        weight = float(component_weights.get(key, 0.0))
        if weight <= 0:
            continue
        weighted_sum += float(score) * weight
        weight_total += weight
    overall_score = round(weighted_sum / weight_total, 2) if weight_total > 0 else 0.0

    quality_flags = set(quality_analysis.get("flags", []) or [])
    quality_status = str(quality_analysis.get("status", "unknown")).lower()
    quality_trend = str(quality_analysis.get("trend", "unknown")).lower()
    speed_status = str(speed_analysis.get("status", "unknown")).lower()
    error_severity = str(error_analysis.get("severity", "none")).lower()
    alert_confidence_by_category: Dict[str, float] = {}
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        alert_category = str(alert.get("category") or "").strip().lower()
        if not alert_category:
            continue
        if alert_category == "error":
            alert_category = "errors"

        alert_confidence = _to_float(alert.get("confidence"))
        if alert_confidence is None:
            alert_severity = str(alert.get("severity") or "").strip().lower()
            if alert_severity == "critical":
                alert_confidence = 0.9
            elif alert_severity == "warning":
                alert_confidence = 0.72
            else:
                continue
        alert_confidence = round(_clamp(alert_confidence, 0.1, 0.99), 2)
        prior_confidence = alert_confidence_by_category.get(alert_category)
        if prior_confidence is None or alert_confidence > prior_confidence:
            alert_confidence_by_category[alert_category] = alert_confidence

    bottlenecks: List[Dict[str, Any]] = []
    if "loss_nan_detected" in quality_flags or "loss_inf_detected" in quality_flags:
        bottlenecks.append({
            "code": "QUALITY_NUMERICAL",
            "severity": "critical",
            "message": "NaN/Inf loss detected. Continue training only after stabilization.",
            "evidence": {"quality_flags": sorted(quality_flags)},
        })
    elif quality_status == "critical":
        bottlenecks.append({
            "code": "QUALITY_CRITICAL",
            "severity": "critical",
            "message": "Quality diagnostics indicate critical instability.",
            "evidence": {"quality_status": quality_status},
        })
    elif quality_trend == "regressing":
        bottlenecks.append({
            "code": "QUALITY_REGRESSION",
            "severity": "warning",
            "message": "Loss trend is regressing; current hyperparameters may be too aggressive.",
            "evidence": {"trend": quality_trend, "improvement_pct": quality_analysis.get("improvement_pct")},
        })

    speed_threshold = float(adaptive_thresholds.get("speed_bottleneck_score", 50.0))
    speed_is_bottleneck = False
    speed_bottleneck_severity = "warning"
    if speed_status == "critical":
        speed_is_bottleneck = True
        speed_bottleneck_severity = "critical"
    elif speed_status == "slow":
        speed_is_bottleneck = speed_score < speed_threshold
    elif speed_score < speed_threshold:
        speed_is_bottleneck = True

    if speed_is_bottleneck:
        bottlenecks.append({
            "code": "THROUGHPUT_LOW",
            "severity": speed_bottleneck_severity,
            "message": "Observed throughput is below the adaptive healthy range for this run profile.",
            "evidence": {
                "speed_status": speed_status,
                "mean_iter_per_sec": (speed_analysis.get("iter_per_sec") or {}).get("mean"),
                "speed_score": round(speed_score, 2),
                "adaptive_speed_threshold": speed_threshold,
                "model_family": model_family,
                "run_class": run_class,
            },
        })

    if error_severity == "critical":
        bottlenecks.append({
            "code": "ERROR_CRITICAL",
            "severity": "critical",
            "message": "Critical runtime errors detected in logs.",
            "evidence": {
                "total_mentions": error_analysis.get("total_mentions"),
                "dominant_category": error_analysis.get("dominant_category"),
            },
        })
    elif error_severity == "warning":
        bottlenecks.append({
            "code": "ERROR_WARNING",
            "severity": "warning",
            "message": "Warning-level runtime errors are present in logs.",
            "evidence": {
                "total_mentions": error_analysis.get("total_mentions"),
                "dominant_category": error_analysis.get("dominant_category"),
            },
        })

    if dataset_analysis.get("exists") is False:
        bottlenecks.append({
            "code": "DATASET_UNRESOLVED",
            "severity": "warning",
            "message": "Dataset diagnostics could not be resolved for this run.",
            "evidence": {"dataset_error": dataset_analysis.get("error")},
        })
    elif dataset_score is not None and dataset_score < 70:
        bottlenecks.append({
            "code": "DATASET_QUALITY_LOW",
            "severity": "warning",
            "message": "Dataset quality score is low and likely limiting results.",
            "evidence": {
                "dataset_quality_score": dataset_score,
                "caption_coverage_pct": dataset_analysis.get("caption_coverage_pct"),
                "duplicate_image_estimate": dataset_analysis.get("duplicate_image_estimate"),
            },
        })

    style_flags = set(nlp_analysis.get("style_flags", []) or [])
    if caption_score is not None and (caption_score < 65 or style_flags):
        bottlenecks.append({
            "code": "CAPTION_QUALITY_LOW",
            "severity": "warning",
            "message": "Caption quality/diversity may be reducing prompt alignment.",
            "evidence": {
                "caption_quality_score": caption_score,
                "caption_token_diversity": nlp_analysis.get("caption_token_diversity"),
                "style_flags": sorted(style_flags),
            },
        })

    bottleneck_category_by_code = {
        "QUALITY_NUMERICAL": "quality",
        "QUALITY_CRITICAL": "quality",
        "QUALITY_REGRESSION": "quality",
        "THROUGHPUT_LOW": "speed",
        "ERROR_CRITICAL": "errors",
        "ERROR_WARNING": "errors",
        "DATASET_UNRESOLVED": "dataset",
        "DATASET_QUALITY_LOW": "dataset",
        "CAPTION_QUALITY_LOW": "nlp",
    }
    for bottleneck in bottlenecks:
        code = str(bottleneck.get("code") or "")
        severity = str(bottleneck.get("severity") or "warning").lower()
        category = bottleneck_category_by_code.get(code)
        confidence = alert_confidence_by_category.get(category) if category else None
        confidence_source = "alert_category_max" if confidence is not None else "severity_default"
        if confidence is None:
            if severity == "critical":
                confidence = 0.9
            elif severity == "warning":
                confidence = 0.72
            else:
                confidence = 0.65
        bottleneck["confidence"] = round(_clamp(float(confidence), 0.1, 0.99), 2)
        bottleneck["confidence_source"] = confidence_source

    recommendation_map = {
        "QUALITY_NUMERICAL": "Reduce learning rate by 20-30% and relaunch with `run-training-with-guardrails`.",
        "QUALITY_CRITICAL": "Pause and stabilize loss signals before extending training steps.",
        "QUALITY_REGRESSION": "Lower learning rate or increase effective batch stability before next run.",
        "THROUGHPUT_LOW": "Lower resolution or optimize batch/VRAM settings to recover throughput.",
        "ERROR_CRITICAL": "Address runtime error root cause before continuing this job.",
        "ERROR_WARNING": "Inspect warning categories and harden environment/dependencies.",
        "DATASET_UNRESOLVED": "Pass explicit `dataset_path` or `config_name` so dataset diagnostics can run.",
        "DATASET_QUALITY_LOW": "Run `auto-improve-dataset` to fix captions/duplicates/corrupt files.",
        "CAPTION_QUALITY_LOW": "Enrich caption detail and diversity; avoid repeated short captions.",
    }
    recommendations: List[Dict[str, Any]] = []
    seen_codes = set()
    for bottleneck in bottlenecks:
        code = str(bottleneck.get("code") or "")
        if not code or code in seen_codes:
            continue
        if code not in recommendation_map:
            continue
        seen_codes.add(code)
        recommendations.append({
            "priority": bottleneck.get("severity", "warning"),
            "for": code,
            "action": recommendation_map[code],
        })

    has_critical_alert = any(str(alert.get("severity", "")).lower() == "critical" for alert in alerts)
    requires_immediate_stop = any(
        str(item.get("code") or "") in {"QUALITY_NUMERICAL", "QUALITY_CRITICAL", "ERROR_CRITICAL"}
        for item in bottlenecks
    )
    dataset_repair_threshold = float(adaptive_thresholds.get("dataset_repair_score", 70.0))
    caption_repair_threshold = float(adaptive_thresholds.get("caption_repair_score", 65.0))
    ready_threshold = float(adaptive_thresholds.get("ready_overall_min", 75.0))
    safe_export_threshold = float(adaptive_thresholds.get("safe_export_overall_min", ready_threshold))

    requires_dataset_repair = bool(dataset_score is not None and dataset_score < dataset_repair_threshold)
    requires_caption_repair = bool(caption_score is not None and caption_score < caption_repair_threshold)
    overall_health = str(snapshot.get("overall_health", "unknown")).lower()

    eval_passed = bool(evaluation.get("passed")) if isinstance(evaluation, dict) and "passed" in evaluation else None
    ready_for_long_run = bool(
        overall_score >= ready_threshold
        and not requires_immediate_stop
        and not has_critical_alert
        and overall_health != "critical"
    )
    safe_to_export = bool(
        ready_for_long_run
        and overall_score >= safe_export_threshold
        and (eval_passed is not False)
    )

    observed_components = sum(1 for val in component_scores.values() if val is not None)
    confidence = round(_clamp(0.35 + (observed_components * 0.12), 0.35, 0.95), 2)

    return {
        "scores": {
            "overall": overall_score,
            "components": component_scores,
            "weights": component_weights,
        },
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "decision_gates": {
            "ready_for_long_run": ready_for_long_run,
            "requires_immediate_stop": requires_immediate_stop,
            "requires_dataset_repair": requires_dataset_repair,
            "requires_caption_repair": requires_caption_repair,
            "safe_to_export": safe_to_export,
        },
        "thresholds": {
            "ready_overall_min": ready_threshold,
            "safe_export_overall_min": safe_export_threshold,
            "speed_bottleneck_score": speed_threshold,
            "dataset_repair_score": dataset_repair_threshold,
            "caption_repair_score": caption_repair_threshold,
        },
        "context": {
            "model_family": model_family,
            "run_class": run_class,
            "hparams": {k: v for k, v in hparams.items() if v is not None},
            "speed_curve_thresholds": _speed_curve_thresholds(model_family, run_class),
        },
        "confidence": confidence,
    }


def summarize_ranking_position(
    ranking_payload: Dict[str, Any],
    target_job_id: str,
) -> Dict[str, Any]:
    """Summarize target job rank/percentile from rank_run_comparisons output."""
    ranking = ranking_payload.get("ranking", []) if isinstance(ranking_payload.get("ranking"), list) else []
    if not ranking:
        return {
            "found": False,
            "job_id": target_job_id,
            "rank": None,
            "total": 0,
            "percentile": None,
            "score": None,
        }

    target = None
    target_index: Optional[int] = None
    for idx, entry in enumerate(ranking):
        if str(entry.get("job_id", "")) == str(target_job_id):
            target = entry
            target_index = idx
            break

    if target is None:
        return {
            "found": False,
            "job_id": target_job_id,
            "rank": None,
            "total": len(ranking),
            "percentile": None,
            "score": None,
        }

    parsed_rank = _to_int(target.get("rank"))
    total = len(ranking)
    if parsed_rank is None or parsed_rank < 1 or parsed_rank > total:
        rank = (target_index + 1) if target_index is not None else 1
        rank_source = "computed_position"
    else:
        rank = parsed_rank
        rank_source = "reported_rank"
    percentile = 100.0 if total <= 1 else round(((total - rank) / (total - 1)) * 100.0, 2)
    score = _to_float((target.get("scores") or {}).get("total"))

    return {
        "found": True,
        "job_id": target_job_id,
        "rank": rank,
        "total": total,
        "percentile": percentile,
        "score": round(score, 2) if score is not None else None,
        "rank_source": rank_source,
        "winner_job_id": ranking_payload.get("winner_job_id"),
        "winner_score": ranking_payload.get("winner_score"),
        "score_spread": ranking_payload.get("score_spread"),
    }


def _extract_hparams_from_config(config_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract core hyperparameters from config payload."""
    if not config_payload:
        return {}
    cfg = config_payload.get("config", config_payload)
    if not isinstance(cfg, dict):
        return {}
    process = cfg.get("process", [])
    if not isinstance(process, list) or not process or not isinstance(process[0], dict):
        return {}
    p0 = process[0]
    train_cfg = p0.get("train") if isinstance(p0.get("train"), dict) else {}
    network_cfg = p0.get("network") if isinstance(p0.get("network"), dict) else {}
    model_cfg = p0.get("model") if isinstance(p0.get("model"), dict) else {}
    datasets = p0.get("datasets")
    ds0 = datasets[0] if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict) else {}
    resolution_values = _normalize_resolution_values(ds0.get("resolution"))

    return {
        "learning_rate": _to_float(train_cfg.get("lr")),
        "steps": _to_int(train_cfg.get("steps")),
        "batch_size": _to_int(train_cfg.get("batch_size")),
        "rank": _to_int(network_cfg.get("linear")),
        "alpha": _to_int(network_cfg.get("linear_alpha")),
        "low_vram": bool(model_cfg.get("low_vram", False)),
        "resolution": max(resolution_values) if resolution_values else None,
    }


def suggest_next_experiment_from_run(
    run_report: Dict[str, Any],
    evaluation: Dict[str, Any],
    config_payload: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Recommend the single highest-value next experiment for a run."""
    metrics = run_report.get("metrics", {}) if isinstance(run_report, dict) else {}
    quality = run_report.get("quality_analysis", {}) if isinstance(run_report, dict) else {}
    error = run_report.get("error_analysis", {}) if isinstance(run_report, dict) else {}
    dataset = run_report.get("dataset_analysis", {}) if isinstance(run_report, dict) else {}
    nlp = run_report.get("nlp_analysis", {}) if isinstance(run_report, dict) else {}
    hparams = _extract_hparams_from_config(config_payload)

    convergence = _to_float((evaluation.get("components") or {}).get("convergence")) or 50.0
    stability = _to_float((evaluation.get("components") or {}).get("stability")) or 50.0
    speed = _to_float((evaluation.get("components") or {}).get("speed"))
    dataset_score = _to_float((evaluation.get("components") or {}).get("dataset"))
    caption_score = _to_float((evaluation.get("components") or {}).get("caption_quality"))

    rec_type = "fine_tune"
    param = None
    current = None
    proposed = None
    expected_impact = "moderate"
    risk_level = "medium"
    rationale = ""

    if dataset_score is not None and dataset_score < 70:
        rec_type = "dataset"
        param = "dataset_curation"
        current = {
            "caption_coverage_pct": dataset.get("caption_coverage_pct"),
            "duplicate_image_estimate": dataset.get("duplicate_image_estimate"),
        }
        proposed = {
            "caption_coverage_pct_target": 95,
            "deduplicate": True,
            "remove_corrupt_images": True,
        }
        expected_impact = "high"
        risk_level = "low"
        rationale = "Dataset quality is the primary bottleneck."
    elif stability < 60:
        oom_count = int((error.get("category_counts") or {}).get("oom", 0) or 0)
        has_nan = "loss_nan_detected" in set(quality.get("flags", []) or [])
        if has_nan:
            rec_type = "stability"
            param = "learning_rate"
            current = hparams.get("learning_rate")
            proposed = round(max(1e-6, (current or 2e-4) * 0.7), 8)
            expected_impact = "high"
            risk_level = "low"
            rationale = "Numerical instability detected; reducing LR is the safest first adjustment."
        elif oom_count > 0:
            rec_type = "memory"
            param = "batch_size"
            current = hparams.get("batch_size")
            proposed = max(1, int((current or 1) - 1))
            expected_impact = "high"
            risk_level = "low"
            rationale = "Repeated OOM signals indicate memory pressure."
        else:
            rec_type = "stability"
            param = "learning_rate"
            current = hparams.get("learning_rate")
            proposed = round(max(1e-6, (current or 2e-4) * 0.8), 8)
            expected_impact = "moderate"
            risk_level = "low"
            rationale = "Stability is below target; conservative LR reduction is recommended."
    elif convergence < 65:
        steps_current = hparams.get("steps")
        lr_current = hparams.get("learning_rate")
        if steps_current is not None and steps_current < 3000:
            rec_type = "convergence"
            param = "steps"
            current = steps_current
            proposed = int(round(steps_current * 1.35))
            expected_impact = "moderate"
            risk_level = "low"
            rationale = "Convergence is weak and total steps are low."
        elif lr_current is not None and lr_current > 3e-4:
            rec_type = "convergence"
            param = "learning_rate"
            current = lr_current
            proposed = round(max(1e-6, lr_current * 0.75), 8)
            expected_impact = "moderate"
            risk_level = "low"
            rationale = "High learning rate may be preventing convergence."
        else:
            rec_type = "capacity"
            param = "rank"
            current = hparams.get("rank")
            proposed = min(128, int((current or 16) + 8))
            expected_impact = "moderate"
            risk_level = "medium"
            rationale = "Convergence plateau suggests LoRA capacity may be low."
    elif speed is not None and speed < 50:
        rec_type = "throughput"
        param = "resolution"
        current = hparams.get("resolution")
        if current is None:
            current = 1024
        proposed = max(512, int(round(current * 0.85)))
        expected_impact = "moderate"
        risk_level = "medium"
        rationale = "Run quality is acceptable but throughput is limiting iteration velocity."
    elif caption_score is not None and caption_score < 65:
        rec_type = "captioning"
        param = "caption_quality"
        current = {"token_diversity": nlp.get("caption_token_diversity"), "style_flags": nlp.get("style_flags")}
        proposed = {"add_detail": True, "increase_caption_diversity": True}
        expected_impact = "moderate"
        risk_level = "low"
        rationale = "Caption quality is limiting prompt alignment/generalization."
    else:
        rec_type = "exploration"
        param = "seeded_eval_prompts"
        current = None
        proposed = {"expand_prompt_suite": True, "add_out_of_distribution_prompts": True}
        expected_impact = "moderate"
        risk_level = "low"
        rationale = "Current run is healthy; prioritize broader evaluation coverage next."

    return {
        "recommendation_type": rec_type,
        "parameter": param,
        "current": current,
        "proposed": proposed,
        "expected_impact": expected_impact,
        "risk_level": risk_level,
        "rationale": rationale,
        "validation_plan": [
            "Run `validate-training-config` after applying the parameter change.",
            "Launch with `run-training-with-guardrails` and monitor first 5-10 polls.",
            "Compare the new run with `compare-training-runs` before promoting.",
        ],
    }


async def collect_job_run_report(
    client: "AIToolkitClient",
    job_id: str,
    *,
    log_lines: int = 1000,
    include_dataset: bool = True,
    include_nlp: bool = True,
    max_dataset_files: int = 2000,
    dataset_cache: Optional[Dict[str, Tuple[Dict[str, Any], List[str]]]] = None,
) -> Dict[str, Any]:
    """Collect structured diagnostics for one training run."""
    if dataset_cache is None:
        dataset_cache = {}

    status_snapshot = await client.get_training_status(job_id)
    if not status_snapshot.get("found"):
        return {"found": False, "error": "job_not_found", "job_id": job_id}

    job_payload = await client.get_training_job(job_id)
    job_data = job_payload.get("job", {}) if job_payload.get("found") else {}
    logs_response = await client.get_training_logs(job_id, log_lines)
    log_text = logs_response.get("log", "") if logs_response.get("success") else ""

    speed_analysis = analyze_speed_from_logs(log_text, speed_string=status_snapshot.get("speed_string", ""))
    quality_analysis = analyze_quality_from_logs(log_text)
    error_analysis = analyze_error_signals(log_text)

    config_payload = _parse_structured_config(job_data.get("job_config")) if job_data else None
    dataset_path_value = _extract_dataset_path_from_config(config_payload or {})
    dataset_analysis = None
    caption_texts: List[str] = []
    if include_dataset or include_nlp:
        if dataset_path_value:
            cache_key = str(dataset_path_value)
            if cache_key not in dataset_cache:
                dataset_cache[cache_key] = analyze_dataset_folder(
                    Path(str(dataset_path_value)),
                    max_files=max_dataset_files
                )
            dataset_analysis, caption_texts = dataset_cache[cache_key]
        else:
            dataset_analysis = {
                "dataset_path": None,
                "exists": False,
                "error": "dataset_path_unresolved"
            }

    nlp_analysis = analyze_nlp_signals(log_text, caption_texts) if include_nlp else {"skipped": True}
    alerts = build_observability_alerts(
        speed_analysis=speed_analysis,
        quality_analysis=quality_analysis,
        error_analysis=error_analysis,
        dataset_analysis=dataset_analysis if include_dataset else None,
        nlp_analysis=nlp_analysis if include_nlp else None,
    )
    health = overall_health_from_alerts(alerts)
    stability_score = score_stability(quality_analysis, error_analysis, alerts)
    dataset_quality_score = score_dataset_quality(dataset_analysis) if include_dataset else None

    run_report = {
        "job_id": job_id,
        "job_name": job_data.get("name"),
        "status": status_snapshot.get("status"),
        "overall_health": health,
        "metrics": {
            "convergence_score": quality_analysis.get("convergence_score"),
            "speed_mean_iter_per_sec": (speed_analysis.get("iter_per_sec") or {}).get("mean"),
            "speed_last_iter_per_sec": (speed_analysis.get("iter_per_sec") or {}).get("last"),
            "stability_score": stability_score,
            "dataset_quality_score": dataset_quality_score,
        },
        "quality_analysis": quality_analysis,
        "speed_analysis": speed_analysis,
        "error_analysis": error_analysis,
        "dataset_analysis": dataset_analysis if include_dataset else {"skipped": True},
        "nlp_analysis": nlp_analysis if include_nlp else {"skipped": True},
        "alerts": alerts,
        "highlights": [alert.get("message") for alert in alerts[:5]],
        "config_payload": config_payload,
        "dataset_path": dataset_path_value,
    }
    return {"found": True, "run_report": run_report}


def inspect_dataset_files(dataset_path: Path, max_files: int = 5000) -> Dict[str, Any]:
    """Inspect dataset files and collect actionable file-level findings."""
    if not dataset_path.exists() or not dataset_path.is_dir():
        return {
            "dataset_path": str(dataset_path),
            "exists": False,
            "missing_caption_files": [],
            "empty_caption_files": [],
            "duplicate_files": [],
            "corrupt_files": [],
            "scanned_images": 0,
        }

    image_files = [
        p for p in sorted(dataset_path.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ][:max_files]

    missing_caption_files: List[str] = []
    empty_caption_files: List[str] = []
    duplicate_files: List[str] = []
    corrupt_files: List[str] = []
    signature_first_seen: Dict[str, str] = {}

    for image_path in image_files:
        try:
            with open(image_path, "rb") as f:
                blob = f.read()
            signature = f"{len(blob)}:{hashlib.md5(blob[:8192]).hexdigest()}"
            if signature in signature_first_seen:
                duplicate_files.append(image_path.name)
            else:
                signature_first_seen[signature] = image_path.name

            with Image.open(image_path) as img:
                _ = img.size
        except Exception:
            corrupt_files.append(image_path.name)

        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            missing_caption_files.append(image_path.name)
        else:
            caption = _safe_read_text(caption_path).strip()
            if not caption:
                empty_caption_files.append(image_path.name)

    return {
        "dataset_path": str(dataset_path),
        "exists": True,
        "missing_caption_files": missing_caption_files,
        "empty_caption_files": empty_caption_files,
        "duplicate_files": duplicate_files,
        "corrupt_files": corrupt_files,
        "scanned_images": len(image_files),
    }


def suggest_caption_from_filename(image_path: Path, trigger_word: Optional[str] = None) -> str:
    """Generate a simple fallback caption from filename tokens."""
    stem = image_path.stem.lower()
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", stem) if tok and not tok.isdigit()]
    tokens = [tok for tok in tokens if tok not in {"img", "image", "photo", "pic", "screenshot", "copy"}]
    token_text = " ".join(tokens[:8]).strip()
    if not token_text:
        token_text = "training image"
    if trigger_word:
        return f"{trigger_word} {token_text}".strip()
    return token_text


def _resolve_unique_path(path: Path) -> Path:
    """Resolve a unique path by appending incremental suffixes when needed."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _move_to_folder(source: Path, target_dir: Path) -> Optional[str]:
    """Move a file into target_dir, preserving filename with collision handling."""
    if not source.exists() or not source.is_file():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = _resolve_unique_path(target_dir / source.name)
    source.rename(destination)
    return str(destination)


def apply_dataset_improvements(
    dataset_path: Path,
    inspection: Dict[str, Any],
    *,
    trigger_word: Optional[str],
    fill_missing_captions: bool,
    fill_empty_captions: bool,
    quarantine_duplicates: bool,
    quarantine_corrupt: bool,
) -> Dict[str, Any]:
    """Apply safe dataset improvements based on inspection findings."""
    actions = {
        "captions_created": 0,
        "empty_captions_filled": 0,
        "duplicates_quarantined": 0,
        "corrupt_quarantined": 0,
        "moved_files": [],
    }

    missing_files = inspection.get("missing_caption_files", []) or []
    empty_files = inspection.get("empty_caption_files", []) or []
    duplicate_files = inspection.get("duplicate_files", []) or []
    corrupt_files = inspection.get("corrupt_files", []) or []

    if fill_missing_captions:
        for image_name in missing_files:
            image_path = dataset_path / image_name
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                continue
            caption_text = suggest_caption_from_filename(image_path, trigger_word=trigger_word)
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption_text + "\n")
            actions["captions_created"] += 1

    if fill_empty_captions:
        for image_name in empty_files:
            image_path = dataset_path / image_name
            caption_path = image_path.with_suffix(".txt")
            caption_text = suggest_caption_from_filename(image_path, trigger_word=trigger_word)
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption_text + "\n")
            actions["empty_captions_filled"] += 1

    if quarantine_duplicates:
        duplicate_dir = dataset_path / "_duplicates"
        for image_name in duplicate_files:
            image_path = dataset_path / image_name
            moved_image = _move_to_folder(image_path, duplicate_dir)
            if moved_image:
                actions["duplicates_quarantined"] += 1
                actions["moved_files"].append({"type": "duplicate_image", "path": moved_image})
                caption_path = image_path.with_suffix(".txt")
                moved_caption = _move_to_folder(caption_path, duplicate_dir)
                if moved_caption:
                    actions["moved_files"].append({"type": "duplicate_caption", "path": moved_caption})

    if quarantine_corrupt:
        quarantine_dir = dataset_path / "_quarantine"
        for image_name in corrupt_files:
            image_path = dataset_path / image_name
            moved_image = _move_to_folder(image_path, quarantine_dir)
            if moved_image:
                actions["corrupt_quarantined"] += 1
                actions["moved_files"].append({"type": "corrupt_image", "path": moved_image})
                caption_path = image_path.with_suffix(".txt")
                moved_caption = _move_to_folder(caption_path, quarantine_dir)
                if moved_caption:
                    actions["moved_files"].append({"type": "corrupt_caption", "path": moved_caption})

    return actions


def _alert_fingerprint(alert: Dict[str, Any]) -> str:
    """Create a stable fingerprint for alert dedupe decisions."""
    severity = str(alert.get("severity") or "").strip().lower()
    category = str(alert.get("category") or "").strip().lower()
    code = str(alert.get("code") or "").strip().lower()
    message = str(alert.get("message") or "").strip().lower()
    raw = f"{severity}|{category}|{code}|{message}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_alert_routing_state(state_path: Path) -> Dict[str, Any]:
    """Load persisted alert-routing state."""
    if not state_path.exists() or not state_path.is_file():
        return {"recent": {}, "window_start_ts": 0.0, "window_count": 0}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {"recent": {}, "window_start_ts": 0.0, "window_count": 0}
        recent = payload.get("recent", {})
        if not isinstance(recent, dict):
            recent = {}
        filtered_recent = {
            str(key): float(value)
            for key, value in recent.items()
            if isinstance(value, (int, float))
        }
        return {
            "recent": filtered_recent,
            "window_start_ts": float(payload.get("window_start_ts", 0.0) or 0.0),
            "window_count": int(payload.get("window_count", 0) or 0),
        }
    except Exception:
        return {"recent": {}, "window_start_ts": 0.0, "window_count": 0}


def save_alert_routing_state(state_path: Path, state: Dict[str, Any]) -> None:
    """Persist alert-routing state atomically."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    payload = {
        "recent": state.get("recent", {}),
        "window_start_ts": state.get("window_start_ts", 0.0),
        "window_count": state.get("window_count", 0),
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(state_path)


def filter_alerts_for_routing(
    alerts: List[Dict[str, Any]],
    *,
    dedupe_window_seconds: int,
    rate_limit_window_seconds: int,
    max_alerts_per_window: int,
    state: Optional[Dict[str, Any]] = None,
    now_ts: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    """Apply dedupe + rate limit filters and return updated routing state."""
    current_ts = float(now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp())
    dedupe_window = max(0, int(dedupe_window_seconds))
    rate_window = max(1, int(rate_limit_window_seconds))
    max_per_window = max(0, int(max_alerts_per_window))

    mutable_state = dict(state or {})
    recent_raw = mutable_state.get("recent", {})
    recent: Dict[str, float] = {}
    if isinstance(recent_raw, dict):
        recent = {
            str(key): float(value)
            for key, value in recent_raw.items()
            if isinstance(value, (int, float))
        }

    if dedupe_window > 0:
        recent = {
            key: ts for key, ts in recent.items()
            if current_ts - ts < dedupe_window
        }
    else:
        recent = {}

    window_start_ts = float(mutable_state.get("window_start_ts", 0.0) or 0.0)
    window_count = int(mutable_state.get("window_count", 0) or 0)
    if window_start_ts <= 0 or current_ts - window_start_ts >= rate_window:
        window_start_ts = current_ts
        window_count = 0

    routed: List[Dict[str, Any]] = []
    dropped_duplicate = 0
    dropped_rate_limit = 0

    for alert in alerts:
        if not isinstance(alert, dict):
            continue

        fingerprint = _alert_fingerprint(alert)
        if dedupe_window > 0:
            previous_ts = recent.get(fingerprint)
            if previous_ts is not None and current_ts - previous_ts < dedupe_window:
                dropped_duplicate += 1
                continue

        if max_per_window > 0 and window_count >= max_per_window:
            dropped_rate_limit += 1
            continue

        routed.append(alert)
        window_count += 1
        if dedupe_window > 0:
            recent[fingerprint] = current_ts

    next_state = {
        "recent": recent,
        "window_start_ts": window_start_ts,
        "window_count": window_count,
    }
    stats = {
        "input_alerts": len(alerts),
        "routed_alerts": len(routed),
        "dropped_duplicate": dropped_duplicate,
        "dropped_rate_limit": dropped_rate_limit,
    }
    return routed, stats, next_state


def _sanitize_retry_status_codes(raw_codes: Any) -> List[int]:
    """Normalize retry status code values into a sorted unique list."""
    if not isinstance(raw_codes, list):
        return list(ALERT_ROUTING_DEFAULT_RETRY_STATUS_CODES)

    normalized: List[int] = []
    for value in raw_codes:
        try:
            code = int(value)
        except (TypeError, ValueError):
            continue
        if 100 <= code <= 599:
            normalized.append(code)

    if not normalized:
        return list(ALERT_ROUTING_DEFAULT_RETRY_STATUS_CODES)
    return sorted(set(normalized))


def _is_retryable_webhook_status(status_code: int, retry_status_codes: List[int]) -> bool:
    """Check if a webhook response status should trigger retry."""
    return int(status_code) in set(retry_status_codes)


def _compute_retry_sleep_seconds(attempt_number: int, base_backoff_ms: int, multiplier: float) -> float:
    """Compute bounded exponential backoff sleep in seconds for retry attempts."""
    base_ms = max(0, int(base_backoff_ms))
    factor = max(1.0, float(multiplier))
    if base_ms <= 0:
        return 0.0
    # attempt_number starts at 1 for the first retry delay slot.
    delay_ms = base_ms * (factor ** max(0, attempt_number - 1))
    bounded_ms = min(120_000.0, delay_ms)
    return bounded_ms / 1000.0


async def send_webhook_with_retry(
    destination: str,
    payload: Dict[str, Any],
    *,
    retry_attempts: int,
    base_backoff_ms: int,
    backoff_multiplier: float,
    retry_status_codes: List[int],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Send webhook payload with retry/backoff on transient failures."""
    attempts = max(1, int(retry_attempts))
    timeout_value = max(1, int(timeout_seconds))
    statuses = _sanitize_retry_status_codes(retry_status_codes)
    history: List[Dict[str, Any]] = []
    last_status: Optional[int] = None
    last_error: Optional[str] = None

    for attempt in range(1, attempts + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(str(destination), json=payload, timeout=timeout_value) as resp:
                    last_status = int(resp.status)
                    response_text = await resp.text() if resp.status >= 400 else None
                    history.append({"attempt": attempt, "status": last_status})

                    if resp.status < 400:
                        return {
                            "success": True,
                            "status": last_status,
                            "error": None,
                            "attempts": attempt,
                            "history": history,
                        }

                    last_error = response_text or f"HTTP {last_status}"
                    if attempt < attempts and _is_retryable_webhook_status(last_status, statuses):
                        await asyncio.sleep(
                            _compute_retry_sleep_seconds(
                                attempt_number=attempt,
                                base_backoff_ms=base_backoff_ms,
                                multiplier=backoff_multiplier,
                            )
                        )
                        continue

                    return {
                        "success": False,
                        "status": last_status,
                        "error": last_error,
                        "attempts": attempt,
                        "history": history,
                    }
        except Exception as exc:
            last_error = str(exc)
            history.append({"attempt": attempt, "exception": last_error})
            if attempt < attempts:
                await asyncio.sleep(
                    _compute_retry_sleep_seconds(
                        attempt_number=attempt,
                        base_backoff_ms=base_backoff_ms,
                        multiplier=backoff_multiplier,
                    )
                )
                continue
            return {
                "success": False,
                "status": last_status,
                "error": last_error,
                "attempts": attempt,
                "history": history,
            }

    return {
        "success": False,
        "status": last_status,
        "error": last_error or "webhook_delivery_failed",
        "attempts": attempts,
        "history": history,
    }


async def collect_observability_snapshot(
    client: "AIToolkitClient",
    *,
    job_id: str,
    lines: int,
    config_name: Optional[str],
    dataset_path_override: Optional[str],
    include_dataset: bool,
    include_nlp: bool,
    include_raw_log_excerpt: bool,
    excerpt_lines: int,
    max_dataset_files: int,
) -> Dict[str, Any]:
    """Collect a single observability snapshot (shared by multiple tools)."""
    status_snapshot = await client.get_training_status(job_id)
    job_payload = await client.get_training_job(job_id)
    logs_response = await client.get_training_logs(job_id, lines)
    if not logs_response.get("success"):
        return {
            "success": False,
            "error": logs_response.get("error", "Unknown error"),
            "job_id": job_id,
        }

    log_text = logs_response.get("log", "") or ""
    log_lines = log_text.splitlines()
    speed_string = status_snapshot.get("speed_string", "") if status_snapshot.get("found") else ""

    speed_analysis = analyze_speed_from_logs(log_text, speed_string=speed_string)
    quality_analysis = analyze_quality_from_logs(log_text)
    error_analysis = analyze_error_signals(log_text)

    config_source = None
    config_payload: Optional[Dict[str, Any]] = None
    resolved_config_name = config_name

    if config_name:
        config_payload = load_config(config_name)
        if config_payload:
            config_source = "saved_config"

    if config_payload is None and job_payload.get("found"):
        job_data = job_payload.get("job", {})
        parsed = _parse_structured_config(job_data.get("job_config"))
        if parsed:
            config_payload = parsed
            config_source = "job_payload"
        if not resolved_config_name:
            resolved_config_name = job_data.get("name")

    if config_payload is None and resolved_config_name:
        from_disk = load_config(resolved_config_name)
        if from_disk:
            config_payload = from_disk
            config_source = "saved_config_by_job_name"

    dataset_source = None
    dataset_path_value = dataset_path_override
    if dataset_path_value:
        dataset_source = "argument"
    else:
        inferred_path = _extract_dataset_path_from_config(config_payload or {})
        if inferred_path:
            dataset_path_value = inferred_path
            dataset_source = "config"

    dataset_analysis = None
    caption_texts: List[str] = []
    if include_dataset or include_nlp:
        if dataset_path_value:
            dataset_analysis, caption_texts = analyze_dataset_folder(
                Path(str(dataset_path_value)),
                max_files=max_dataset_files
            )
        else:
            dataset_analysis = {
                "dataset_path": None,
                "exists": False,
                "error": "dataset_path_unresolved"
            }

    nlp_analysis = None
    if include_nlp:
        nlp_analysis = analyze_nlp_signals(log_text, caption_texts)

    alerts = build_observability_alerts(
        speed_analysis=speed_analysis,
        quality_analysis=quality_analysis,
        error_analysis=error_analysis,
        dataset_analysis=dataset_analysis if include_dataset else None,
        nlp_analysis=nlp_analysis if include_nlp else None
    )

    system_snapshot = await client.get_system_stats()
    report = {
        "success": True,
        "job_id": job_id,
        "generated_at": _utc_now_iso(),
        "overall_health": overall_health_from_alerts(alerts),
        "status_snapshot": status_snapshot,
        "system_snapshot": system_snapshot,
        "config_context": {
            "requested_config_name": config_name,
            "resolved_config_name": resolved_config_name,
            "config_source": config_source
        },
        "logs": {
            "analyzed_line_count": len(log_lines),
            "requested_line_count": lines
        },
        "speed_analysis": speed_analysis,
        "quality_analysis": quality_analysis,
        "error_analysis": error_analysis,
        "dataset_analysis": dataset_analysis if include_dataset else {"skipped": True},
        "nlp_analysis": nlp_analysis if include_nlp else {"skipped": True},
        "dataset_resolution": {
            "dataset_path": str(dataset_path_value) if dataset_path_value else None,
            "dataset_source": dataset_source
        },
        "alerts": alerts
    }
    if include_raw_log_excerpt:
        report["raw_log_excerpt"] = "\n".join(log_lines[-excerpt_lines:])
    return report


def _tokenize_text(text: str) -> List[str]:
    """Tokenize text for lightweight NLP-style summaries."""
    if not text:
        return []
    tokens = [tok.lower() for tok in TEXT_TOKEN_PATTERN.findall(text)]
    return [tok for tok in tokens if tok not in STOPWORDS]


def _stat_summary(values: List[float]) -> Dict[str, Any]:
    """Generate stable numeric summary statistics."""
    filtered = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not filtered:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "last": None
        }

    sorted_vals = sorted(filtered)
    return {
        "count": len(sorted_vals),
        "min": round(sorted_vals[0], 6),
        "max": round(sorted_vals[-1], 6),
        "mean": round(statistics.fmean(sorted_vals), 6),
        "median": round(statistics.median(sorted_vals), 6),
        "last": round(sorted_vals[-1], 6)
    }


def _to_float(value: Any) -> Optional[float]:
    """Best-effort parse float values from strings or numerics."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None

    if isinstance(value, str):
        match = FLOAT_PATTERN.search(value)
        if not match:
            return None
        try:
            out = float(match.group(0))
            return out if math.isfinite(out) else None
        except ValueError:
            return None
    return None


def _parse_speed_string_to_iter_per_sec(speed_string: str) -> Optional[float]:
    """Parse AI Toolkit speed text into iteration/sec when possible."""
    if not speed_string:
        return None

    s = speed_string.strip().lower()
    direct = re.search(r"([0-9]*\.?[0-9]+)\s*(?:it/s|iter/s|steps/s|samples/s)", s)
    if direct:
        return _to_float(direct.group(1))

    inv_sec = re.search(r"([0-9]*\.?[0-9]+)\s*(?:s/it|sec/it|seconds/it)", s)
    if inv_sec:
        sec_per_it = _to_float(inv_sec.group(1))
        if sec_per_it and sec_per_it > 0:
            return 1.0 / sec_per_it

    inv_ms = re.search(r"([0-9]*\.?[0-9]+)\s*ms/it", s)
    if inv_ms:
        ms_per_it = _to_float(inv_ms.group(1))
        if ms_per_it and ms_per_it > 0:
            return 1000.0 / ms_per_it
    return None


def analyze_speed_from_logs(log_text: str, speed_string: str = "") -> Dict[str, Any]:
    """Extract speed metrics from logs and AI Toolkit status fields."""
    lines = log_text.splitlines() if log_text else []
    iter_per_sec: List[float] = []
    source_hits = {"it_per_sec": 0, "sec_per_it": 0, "ms_per_it": 0}

    for line in lines:
        lower = line.lower()
        for match in re.finditer(r"([0-9]*\.?[0-9]+)\s*(?:it/s|iter/s|steps/s|samples/s)", lower):
            val = _to_float(match.group(1))
            if val is not None:
                iter_per_sec.append(val)
                source_hits["it_per_sec"] += 1

        for match in re.finditer(r"([0-9]*\.?[0-9]+)\s*(?:s/it|sec/it|seconds/it)", lower):
            val = _to_float(match.group(1))
            if val and val > 0:
                iter_per_sec.append(1.0 / val)
                source_hits["sec_per_it"] += 1

        for match in re.finditer(r"([0-9]*\.?[0-9]+)\s*ms/it", lower):
            val = _to_float(match.group(1))
            if val and val > 0:
                iter_per_sec.append(1000.0 / val)
                source_hits["ms_per_it"] += 1

    status_speed = _parse_speed_string_to_iter_per_sec(speed_string)
    if status_speed is not None:
        iter_per_sec.append(status_speed)

    stats = _stat_summary(iter_per_sec)
    trend = "unknown"
    trend_delta_pct = None
    if stats["count"] and stats["count"] >= 4:
        midpoint = len(iter_per_sec) // 2
        early = statistics.fmean(iter_per_sec[:midpoint]) if midpoint else iter_per_sec[0]
        late = statistics.fmean(iter_per_sec[midpoint:])
        if early > 0:
            trend_delta_pct = round(((late - early) / early) * 100, 2)
            if trend_delta_pct > 5:
                trend = "speeding_up"
            elif trend_delta_pct < -5:
                trend = "slowing_down"
            else:
                trend = "stable"

    status = "unknown"
    if stats["count"] > 0 and stats["mean"] is not None:
        if stats["mean"] < 0.05:
            status = "critical"
        elif stats["mean"] < 0.2:
            status = "slow"
        else:
            status = "healthy"

    return {
        "iter_per_sec": stats,
        "trend": trend,
        "trend_delta_pct": trend_delta_pct,
        "status_speed_string": speed_string or None,
        "status": status,
        "source_hits": source_hits
    }


def analyze_quality_from_logs(log_text: str) -> Dict[str, Any]:
    """Extract quality signals from training logs, mostly around loss convergence."""
    lines = log_text.splitlines() if log_text else []
    losses: List[float] = []
    nan_count = 0
    inf_count = 0

    loss_patterns = [
        re.compile(r"\b(?:loss|train_loss|total_loss|avg_loss|running_loss)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE),
        re.compile(r"\bloss\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    ]

    for line in lines:
        lower = line.lower()
        if "nan" in lower:
            nan_count += 1
        if " inf" in lower or "infinity" in lower:
            inf_count += 1

        for pattern in loss_patterns:
            match = pattern.search(line)
            if not match:
                continue
            val = _to_float(match.group(1))
            if val is not None:
                losses.append(val)
            break

    stats = _stat_summary(losses)
    trend = "unknown"
    improvement_pct = None
    convergence_score = None
    flags: List[str] = []

    if stats["count"] >= 2:
        first = losses[0]
        last = losses[-1]
        if first != 0:
            improvement_pct = round(((first - last) / abs(first)) * 100, 2)
        if improvement_pct is not None:
            if improvement_pct > 10:
                trend = "improving"
            elif improvement_pct < -10:
                trend = "regressing"
            else:
                trend = "flat"

    if nan_count > 0:
        flags.append("loss_nan_detected")
    if inf_count > 0:
        flags.append("loss_inf_detected")
    if stats["count"] == 0:
        flags.append("no_loss_values_detected")
    elif stats["max"] is not None and stats["median"] is not None and stats["median"] > 0:
        if stats["max"] > (stats["median"] * 5):
            flags.append("loss_spikes_detected")

    if stats["count"] > 0:
        convergence_score = 50.0
        if improvement_pct is not None:
            convergence_score += max(min(improvement_pct, 60), -60) * 0.5
        if "loss_spikes_detected" in flags:
            convergence_score -= 20
        if "loss_nan_detected" in flags or "loss_inf_detected" in flags:
            convergence_score = 0
        convergence_score = round(max(0.0, min(100.0, convergence_score)), 2)

    status = "unknown"
    if "loss_nan_detected" in flags or "loss_inf_detected" in flags:
        status = "critical"
    elif trend == "regressing":
        status = "warning"
    elif trend in {"improving", "flat"}:
        status = "healthy"

    return {
        "loss": stats,
        "trend": trend,
        "improvement_pct": improvement_pct,
        "convergence_score": convergence_score,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "flags": flags,
        "status": status
    }


def analyze_error_signals(log_text: str) -> Dict[str, Any]:
    """Detect and classify error/failure patterns in logs."""
    lines = log_text.splitlines() if log_text else []
    categories = {
        "oom": re.compile(r"(out of memory|cuda out of memory|oom)", re.IGNORECASE),
        "traceback": re.compile(r"(traceback|exception|stack trace)", re.IGNORECASE),
        "io": re.compile(r"(file not found|no such file|permission denied|i/o error)", re.IGNORECASE),
        "network": re.compile(r"(timeout|connection reset|connection refused|dns|tls|ssl)", re.IGNORECASE),
        "numerical": re.compile(r"(\bnan\b|\binf\b|overflow|underflow)", re.IGNORECASE),
        "generic_error": re.compile(r"\berror\b", re.IGNORECASE),
    }

    category_counts = {name: 0 for name in categories}
    samples: Dict[str, List[str]] = {name: [] for name in categories}

    for line in lines:
        for cat, pattern in categories.items():
            if not pattern.search(line):
                continue
            category_counts[cat] += 1
            if len(samples[cat]) < 3:
                samples[cat].append(line.strip())

    total_error_mentions = sum(category_counts.values())
    dominant_category = None
    if total_error_mentions > 0:
        dominant_category = max(category_counts.items(), key=lambda item: item[1])[0]

    severity = "none"
    if category_counts["oom"] > 0 or category_counts["traceback"] > 0:
        severity = "critical"
    elif total_error_mentions > 0:
        severity = "warning"

    return {
        "category_counts": category_counts,
        "sample_lines": {k: v for k, v in samples.items() if v},
        "total_mentions": total_error_mentions,
        "dominant_category": dominant_category,
        "severity": severity
    }


def _safe_read_text(path: Path) -> str:
    """Read text from file with forgiving decoding."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _normalize_caption(caption: str) -> str:
    return " ".join((caption or "").strip().lower().split())


def analyze_dataset_folder(dataset_path: Path, max_files: int = 5000) -> Tuple[Dict[str, Any], List[str]]:
    """Analyze dataset integrity, coverage, and lightweight text quality."""
    if not dataset_path.exists():
        return {
            "dataset_path": str(dataset_path),
            "exists": False,
            "error": "dataset_path_not_found"
        }, []

    if not dataset_path.is_dir():
        return {
            "dataset_path": str(dataset_path),
            "exists": False,
            "error": "dataset_path_not_directory"
        }, []

    image_files = [
        p for p in sorted(dataset_path.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    total_images = len(image_files)
    scanned_files = image_files[:max_files]

    widths: List[float] = []
    heights: List[float] = []
    aspect_ratios: List[float] = []
    corrupt_images: List[str] = []
    low_resolution_count = 0
    extreme_aspect_count = 0

    caption_texts: List[str] = []
    caption_missing_count = 0
    caption_empty_count = 0
    unique_caption_set = set()
    signature_counter: Counter[str] = Counter()

    for image_path in scanned_files:
        try:
            with open(image_path, "rb") as f:
                blob = f.read()
            signature = f"{len(blob)}:{hashlib.md5(blob[:8192]).hexdigest()}"
            signature_counter[signature] += 1

            with Image.open(image_path) as img:
                width, height = img.size
            widths.append(float(width))
            heights.append(float(height))
            if height > 0:
                ratio = width / height
                aspect_ratios.append(ratio)
                if ratio > 3.0 or ratio < 0.33:
                    extreme_aspect_count += 1
            if width < 256 or height < 256:
                low_resolution_count += 1
        except Exception:
            corrupt_images.append(image_path.name)

        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            caption_missing_count += 1
            continue
        caption = _safe_read_text(caption_path).strip()
        if not caption:
            caption_empty_count += 1
            continue
        caption_texts.append(caption)
        unique_caption_set.add(_normalize_caption(caption))

    duplicate_image_estimate = sum(max(0, count - 1) for count in signature_counter.values())
    caption_word_counts = [len(c.split()) for c in caption_texts]
    caption_char_counts = [len(c) for c in caption_texts]
    top_caption_terms = Counter()
    for caption in caption_texts:
        top_caption_terms.update(_tokenize_text(caption))

    captions_non_empty = len(caption_texts)
    caption_coverage_pct = round((captions_non_empty / len(scanned_files)) * 100, 2) if scanned_files else 0.0
    unique_caption_ratio = round((len(unique_caption_set) / captions_non_empty), 4) if captions_non_empty else 0.0

    summary = {
        "dataset_path": str(dataset_path),
        "exists": True,
        "total_images": total_images,
        "scanned_images": len(scanned_files),
        "scan_limited": total_images > len(scanned_files),
        "caption_non_empty_count": captions_non_empty,
        "caption_missing_count": caption_missing_count,
        "caption_empty_count": caption_empty_count,
        "caption_coverage_pct": caption_coverage_pct,
        "unique_caption_ratio": unique_caption_ratio,
        "resolution_width": _stat_summary(widths),
        "resolution_height": _stat_summary(heights),
        "aspect_ratio": _stat_summary(aspect_ratios),
        "corrupt_image_count": len(corrupt_images),
        "corrupt_image_examples": corrupt_images[:10],
        "duplicate_image_estimate": duplicate_image_estimate,
        "low_resolution_count": low_resolution_count,
        "extreme_aspect_ratio_count": extreme_aspect_count,
        "caption_words": _stat_summary([float(x) for x in caption_word_counts]),
        "caption_chars": _stat_summary([float(x) for x in caption_char_counts]),
        "top_caption_terms": [
            {"term": term, "count": count}
            for term, count in top_caption_terms.most_common(20)
        ],
        "caption_examples": caption_texts[:10]
    }

    return summary, caption_texts


def analyze_nlp_signals(log_text: str, caption_texts: List[str]) -> Dict[str, Any]:
    """Build lightweight NLP summaries for logs and caption corpus."""
    log_tokens = _tokenize_text(log_text or "")
    caption_tokens = []
    for caption in caption_texts:
        caption_tokens.extend(_tokenize_text(caption))

    log_terms = Counter(log_tokens)
    caption_terms = Counter(caption_tokens)

    total_caption_tokens = len(caption_tokens)
    caption_vocab = len(set(caption_tokens))
    token_diversity = round(caption_vocab / total_caption_tokens, 4) if total_caption_tokens else 0.0

    ascii_chars = sum(sum(1 for ch in caption if ord(ch) < 128) for caption in caption_texts)
    total_chars = sum(len(caption) for caption in caption_texts)
    ascii_ratio = round(ascii_chars / total_chars, 4) if total_chars else None

    style_flags: List[str] = []
    if total_caption_tokens > 0 and token_diversity < 0.2:
        style_flags.append("caption_vocabulary_low_diversity")
    if caption_texts and statistics.fmean([len(c.split()) for c in caption_texts]) < 4:
        style_flags.append("captions_too_short")

    return {
        "log_top_terms": [{"term": term, "count": count} for term, count in log_terms.most_common(25)],
        "caption_top_terms": [{"term": term, "count": count} for term, count in caption_terms.most_common(25)],
        "caption_token_count": total_caption_tokens,
        "caption_vocabulary_size": caption_vocab,
        "caption_token_diversity": token_diversity,
        "caption_ascii_ratio": ascii_ratio,
        "style_flags": style_flags
    }


def _parse_structured_config(config_payload: Any) -> Optional[Dict[str, Any]]:
    """Parse config blobs that might be dict or JSON/YAML string."""
    if config_payload is None:
        return None
    if isinstance(config_payload, dict):
        return config_payload
    if isinstance(config_payload, str):
        text = config_payload.strip()
        if not text:
            return None
        for parser in (json.loads, yaml.safe_load):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _extract_dataset_path_from_config(config: Dict[str, Any]) -> Optional[str]:
    """Extract dataset path from config structures used by AI Toolkit."""
    if not config:
        return None

    metadata = config.get("training_metadata", {})
    if isinstance(metadata, dict) and metadata.get("dataset_path"):
        return str(metadata.get("dataset_path"))

    cfg = config.get("config", config)
    process = cfg.get("process", []) if isinstance(cfg, dict) else []
    if isinstance(process, list) and process:
        first = process[0]
        if isinstance(first, dict):
            datasets = first.get("datasets", [])
            if isinstance(datasets, list) and datasets:
                ds0 = datasets[0]
                if isinstance(ds0, dict):
                    for key in ("folder_path", "dataset_path", "path"):
                        if ds0.get(key):
                            return str(ds0.get(key))
    return None


def build_observability_alerts(
    speed_analysis: Dict[str, Any],
    quality_analysis: Dict[str, Any],
    error_analysis: Dict[str, Any],
    dataset_analysis: Optional[Dict[str, Any]],
    nlp_analysis: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert analysis outputs into actionable alert objects."""
    alerts: List[Dict[str, Any]] = []

    def add_alert(severity: str, category: str, message: str, confidence: Optional[float] = None) -> None:
        if confidence is None:
            base = 0.9 if severity == "critical" else 0.72
            if category == "quality":
                base += 0.05
            elif category == "speed":
                base += 0.03
            elif category == "nlp":
                base -= 0.07
            elif category == "dataset":
                base -= 0.02
            confidence = _clamp(base, 0.55, 0.99)
        alerts.append({
            "severity": severity,
            "category": category,
            "message": message,
            "confidence": round(float(confidence), 2),
        })
    quality_flags = set(quality_analysis.get("flags", []) or [])
    error_category_counts = error_analysis.get("category_counts", {}) or {}
    if not isinstance(error_category_counts, dict):
        error_category_counts = {}
    numerical_mentions = int(error_category_counts.get("numerical", 0) or 0)
    total_error_mentions = int(error_analysis.get("total_mentions", 0) or 0)
    non_numerical_mentions = max(0, total_error_mentions - numerical_mentions)
    quality_nan_or_inf = (
        "loss_nan_detected" in quality_flags or "loss_inf_detected" in quality_flags
    )

    if error_analysis.get("severity") == "critical":
        add_alert(
            "critical",
            "errors",
            f"Critical log errors detected (dominant: {error_analysis.get('dominant_category')}).",
            confidence=0.95,
        )
    elif total_error_mentions > 0:
        # Suppress duplicate warning when numerical-only errors are already captured
        # as critical quality alerts (e.g., `loss: nan`).
        if non_numerical_mentions == 0 and quality_nan_or_inf:
            pass
        else:
            add_alert(
                "warning",
                "errors",
                f"Non-critical error mentions detected ({total_error_mentions}).",
                confidence=0.75,
            )

    if quality_analysis.get("status") == "critical":
        add_alert("critical", "quality", "Loss contains NaN/Inf values.", confidence=0.98)
    elif quality_analysis.get("trend") == "regressing":
        add_alert("warning", "quality", "Loss trend appears to be regressing.", confidence=0.78)

    speed_status = speed_analysis.get("status")
    if speed_status == "critical":
        add_alert("critical", "speed", "Training throughput is critically low.", confidence=0.95)
    elif speed_status == "slow":
        add_alert("warning", "speed", "Training throughput is low.", confidence=0.76)

    if dataset_analysis:
        if not dataset_analysis.get("exists"):
            add_alert(
                "warning",
                "dataset",
                "Dataset path is unavailable; dataset checks were skipped.",
                confidence=0.7,
            )
        else:
            if dataset_analysis.get("caption_coverage_pct", 0) < 90:
                add_alert(
                    "warning",
                    "dataset",
                    f"Caption coverage is low ({dataset_analysis.get('caption_coverage_pct')}%).",
                    confidence=0.74,
                )
            if dataset_analysis.get("duplicate_image_estimate", 0) > 0:
                add_alert(
                    "warning",
                    "dataset",
                    f"Possible duplicate images detected ({dataset_analysis.get('duplicate_image_estimate')}).",
                    confidence=0.76,
                )
            if dataset_analysis.get("corrupt_image_count", 0) > 0:
                add_alert(
                    "critical",
                    "dataset",
                    f"Corrupt images detected ({dataset_analysis.get('corrupt_image_count')}).",
                    confidence=0.94,
                )

    if nlp_analysis:
        for style_flag in nlp_analysis.get("style_flags", []):
            add_alert("warning", "nlp", str(style_flag), confidence=0.66)

    return alerts


def overall_health_from_alerts(alerts: List[Dict[str, str]]) -> str:
    """Summarize overall health from alert severities."""
    severities = {alert.get("severity", "") for alert in alerts}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    return "healthy"

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """Return list of available tools"""
    return [
        types.Tool(
            name="create-training-config",
            description="Create a new LoRA training configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the training job"},
                    "model_name": {"type": "string", "description": "Base model name (e.g., 'black-forest-labs/FLUX.1-dev')"},
                    "dataset_path": {"type": "string", "description": "Path to the dataset folder"},
                    "resolution": {"type": "integer", "description": "Training resolution", "default": 512},
                    "batch_size": {"type": "integer", "description": "Batch size", "default": 1},
                    "learning_rate": {"type": "number", "description": "Learning rate", "default": 0.0002},
                    "steps": {"type": "integer", "description": "Training steps", "default": 1000},
                    "rank": {"type": "integer", "description": "LoRA rank", "default": 16},
                    "alpha": {"type": "integer", "description": "LoRA alpha", "default": 16},
                    "use_wandb": {"type": "boolean", "description": "Enable Weights & Biases logging", "default": False},
                    "low_vram": {"type": "boolean", "description": "Enable low VRAM mode", "default": True},
                    "trigger_word": {"type": "string", "description": "Trigger word for the LoRA"},
                    "test_prompts": {
                        "type": "array", 
                        "description": "Test prompts for validation (4 recommended: 3 similar, 1 unique). All should use the trigger word",
                        "items": {"type": "string"}
                    },
                    "disable_sampling": {"type": "boolean", "description": "Disable sample image generation during training", "default": False}
                },
                "required": ["name", "model_name", "dataset_path"]
            }
        ),
        types.Tool(
            name="validate-training-config",
            description="Validate a training configuration for required fields, model compatibility, and resource sanity checks",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_name": {"type": "string", "description": "Saved configuration name to validate"},
                    "config": {
                        "description": "Raw config payload (dict) or JSON/YAML string",
                        "oneOf": [
                            {"type": "object"},
                            {"type": "string"}
                        ]
                    },
                    "name": {"type": "string", "description": "Draft config name when validating ad-hoc parameters"},
                    "model_name": {"type": "string", "description": "Draft base model name/path for ad-hoc validation"},
                    "dataset_path": {"type": "string", "description": "Draft dataset path for ad-hoc validation"},
                    "resolution": {"type": "integer", "description": "Draft training resolution", "default": 512},
                    "batch_size": {"type": "integer", "description": "Draft batch size", "default": 1},
                    "learning_rate": {"type": "number", "description": "Draft learning rate", "default": 0.0002},
                    "steps": {"type": "integer", "description": "Draft training steps", "default": 1000},
                    "rank": {"type": "integer", "description": "Draft LoRA rank", "default": 16},
                    "alpha": {"type": "integer", "description": "Draft LoRA alpha", "default": 16},
                    "low_vram": {"type": "boolean", "description": "Draft low VRAM mode", "default": True},
                    "trigger_word": {"type": "string", "description": "Draft trigger word"},
                    "test_prompts": {"type": "array", "items": {"type": "string"}, "description": "Draft sample prompts"},
                    "check_dataset": {"type": "boolean", "description": "Run dataset folder diagnostics", "default": True},
                    "max_dataset_files": {"type": "integer", "description": "Max files to scan in dataset diagnostics", "default": 2000}
                },
                "required": []
            }
        ),
        types.Tool(
            name="recommend-training-plan",
            description="Recommend training hyperparameters from goal, dataset profile, time budget, and VRAM target",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Training objective text (style/character/object/etc.)"},
                    "base_model": {"type": "string", "description": "Base model reference (e.g. 'ostris/Flex.1-alpha')"},
                    "model_name": {"type": "string", "description": "Alias for base_model"},
                    "dataset_path": {"type": "string", "description": "Dataset path for diagnostics"},
                    "config_name": {"type": "string", "description": "Optional saved config name to infer dataset/model"},
                    "time_budget_minutes": {"type": "number", "description": "Optional max training time budget (minutes)"},
                    "target_vram_gb": {"type": "number", "description": "Optional target GPU VRAM budget in GB"},
                    "include_dataset_scan": {"type": "boolean", "description": "Scan dataset stats for stronger recommendations", "default": True},
                    "max_dataset_files": {"type": "integer", "description": "Max dataset files to scan", "default": 2000}
                },
                "required": []
            }
        ),
        types.Tool(
            name="run-training-with-guardrails",
            description="Validate config, start training, and auto-stop on critical guardrail conditions",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_name": {"type": "string", "description": "Saved configuration to run", "minLength": 1},
                    "check_dataset": {"type": "boolean", "description": "Run dataset checks in preflight validation", "default": True},
                    "max_dataset_files": {"type": "integer", "description": "Max files scanned for dataset checks", "default": 2000},
                    "block_on_validation_fail": {"type": "boolean", "description": "Block launch when validation has errors", "default": True},
                    "monitor_seconds": {"type": "integer", "description": "Post-launch monitor window in seconds", "default": 120},
                    "poll_interval_seconds": {"type": "integer", "description": "Polling interval during guardrail monitoring", "default": 10},
                    "stop_on_nan": {"type": "boolean", "description": "Stop when NaN/Inf loss is detected", "default": True},
                    "stop_on_repeated_oom": {"type": "boolean", "description": "Stop on repeated OOM errors", "default": True},
                    "oom_threshold": {"type": "integer", "description": "OOM mentions threshold before stop", "default": 3},
                    "stop_on_stalled_progress": {"type": "boolean", "description": "Stop when progress stalls", "default": True},
                    "stall_window_seconds": {"type": "integer", "description": "Seconds without progress before stall", "default": 120},
                    "min_step_delta": {"type": "integer", "description": "Minimum step gain expected during stall window", "default": 1},
                    "stop_on_throughput_collapse": {"type": "boolean", "description": "Stop when throughput collapses", "default": True},
                    "throughput_floor_iter_per_sec": {"type": "number", "description": "Minimum healthy throughput (iter/sec)", "default": 0.02},
                    "low_speed_consecutive_polls": {"type": "integer", "description": "Consecutive low-speed polls before stop", "default": 2},
                    "log_lines": {"type": "integer", "description": "Log tail lines inspected each poll", "default": 400}
                },
                "required": ["config_name"]
            }
        ),
        types.Tool(
            name="watch-training-timeseries",
            description="Poll training status/logs over time and return a structured time-series summary",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                    "duration_seconds": {"type": "integer", "description": "Requested watch window in seconds", "default": 120},
                    "poll_interval_seconds": {"type": "integer", "description": "Polling interval in seconds", "default": 10},
                    "log_lines": {"type": "integer", "description": "Log lines analyzed per poll", "default": 400},
                    "include_system": {"type": "boolean", "description": "Include system (GPU/CPU/RAM) stats per poll", "default": True},
                    "include_alerts": {"type": "boolean", "description": "Include alert objects per point", "default": True},
                    "include_dataset": {"type": "boolean", "description": "Include one-time dataset diagnostics context", "default": False},
                    "include_nlp": {"type": "boolean", "description": "Include lightweight NLP log/caption summaries", "default": False},
                    "dataset_path": {"type": "string", "description": "Optional dataset path override"},
                    "config_name": {"type": "string", "description": "Optional config name for dataset-path resolution"},
                    "stop_on_terminal_status": {"type": "boolean", "description": "Stop watch when job reaches terminal state", "default": True},
                    "include_raw_log_excerpt": {"type": "boolean", "description": "Include raw log tail excerpt in each point", "default": False},
                    "excerpt_lines": {"type": "integer", "description": "Raw excerpt lines when enabled", "default": 80},
                    "max_dataset_files": {"type": "integer", "description": "Max dataset files scanned when include_dataset=true", "default": 2000}
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="compare-training-runs",
            description="Compare multiple training runs across convergence, speed, stability, and dataset quality",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_ids": {"type": "array", "items": {"type": "string"}, "description": "Explicit job IDs to compare"},
                    "limit": {"type": "integer", "description": "When job_ids omitted, compare up to N recent jobs", "default": 5},
                    "log_lines": {"type": "integer", "description": "Log lines analyzed per job", "default": 800},
                    "include_dataset": {"type": "boolean", "description": "Include dataset diagnostics in scoring", "default": True},
                    "include_nlp": {"type": "boolean", "description": "Include NLP diagnostics in run reports", "default": False},
                    "max_dataset_files": {"type": "integer", "description": "Max dataset files scanned per unique dataset", "default": 2000},
                    "only_terminal_runs": {"type": "boolean", "description": "Compare only completed/failed/stopped jobs", "default": False}
                },
                "required": []
            }
        ),
        types.Tool(
            name="evaluate-lora",
            description="Evaluate a LoRA run using a weighted rubric and return component scores + grade",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID to evaluate"},
                    "log_lines": {"type": "integer", "description": "Log lines analyzed for evaluation", "default": 1000},
                    "include_dataset": {"type": "boolean", "description": "Include dataset diagnostics in rubric", "default": True},
                    "include_nlp": {"type": "boolean", "description": "Include NLP/caption diagnostics in rubric", "default": True},
                    "max_dataset_files": {"type": "integer", "description": "Max dataset files scanned", "default": 2000},
                    "pass_threshold": {"type": "number", "description": "Pass threshold for overall score", "default": 70},
                    "rubric_weights": {
                        "type": "object",
                        "description": "Optional weights override: convergence, stability, speed, dataset, caption_quality",
                        "additionalProperties": {"type": "number"}
                    }
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="suggest-next-experiment",
            description="Suggest one highest-value next experiment from one or more evaluated runs",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Primary job ID to analyze"},
                    "job_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional candidate job IDs (best run selected internally)"},
                    "log_lines": {"type": "integer", "description": "Log lines analyzed per run", "default": 1000},
                    "include_dataset": {"type": "boolean", "description": "Include dataset diagnostics", "default": True},
                    "include_nlp": {"type": "boolean", "description": "Include NLP diagnostics", "default": True},
                    "max_dataset_files": {"type": "integer", "description": "Max dataset files scanned", "default": 2000},
                    "pass_threshold": {"type": "number", "description": "Pass threshold used during run evaluation", "default": 70}
                },
                "required": []
            }
        ),
        types.Tool(
            name="auto-improve-dataset",
            description="Analyze dataset issues and optionally apply safe fixes (captions, duplicate/corrupt quarantine)",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_path": {"type": "string", "description": "Dataset directory path"},
                    "config_name": {"type": "string", "description": "Optional config name used to infer dataset path"},
                    "max_dataset_files": {"type": "integer", "description": "Max dataset files inspected", "default": 5000},
                    "apply": {"type": "boolean", "description": "Apply improvements; false returns plan only", "default": False},
                    "trigger_word": {"type": "string", "description": "Optional trigger word used in generated captions"},
                    "fill_missing_captions": {"type": "boolean", "description": "Generate missing caption .txt files", "default": True},
                    "fill_empty_captions": {"type": "boolean", "description": "Overwrite empty captions", "default": True},
                    "quarantine_duplicates": {"type": "boolean", "description": "Move duplicate images to _duplicates", "default": False},
                    "quarantine_corrupt": {"type": "boolean", "description": "Move corrupt images to _quarantine", "default": True}
                },
                "required": []
            }
        ),
        types.Tool(
            name="export-observability-report",
            description="Export observability/evaluation artifacts for a job as JSON + Markdown files",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                    "config_name": {"type": "string", "description": "Optional config name context"},
                    "dataset_path": {"type": "string", "description": "Optional dataset path override"},
                    "output_dir": {"type": "string", "description": "Output directory for report files"},
                    "filename_prefix": {"type": "string", "description": "Prefix for generated report files", "default": "observability"},
                    "lines": {"type": "integer", "description": "Log lines analyzed", "default": 800},
                    "max_dataset_files": {"type": "integer", "description": "Max dataset files scanned", "default": 2000},
                    "include_dataset": {"type": "boolean", "description": "Include dataset diagnostics", "default": True},
                    "include_nlp": {"type": "boolean", "description": "Include NLP diagnostics", "default": True},
                    "include_evaluation": {"type": "boolean", "description": "Attach evaluate-lora rubric output", "default": True},
                    "include_raw_log_excerpt": {"type": "boolean", "description": "Include raw log excerpt", "default": False}
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="alert-routing",
            description="Route alerts to file or webhook sink from provided alerts or derived job observability",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Optional job ID to derive alerts from logs"},
                    "alerts": {"type": "array", "description": "Optional explicit alert objects", "items": {"type": "object"}},
                    "sink": {"type": "string", "description": "Alert sink type", "enum": ["file", "webhook"], "default": "file"},
                    "destination": {"type": "string", "description": "File path or webhook URL destination"},
                    "min_severity": {"type": "string", "description": "Minimum severity routed", "enum": ["info", "warning", "critical"], "default": "warning"},
                    "lines": {"type": "integer", "description": "Log lines used when deriving alerts from job", "default": 500},
                    "include_context": {"type": "boolean", "description": "Include status/context payload with routed alerts", "default": True},
                    "state_path": {"type": "string", "description": "Optional state file path for dedupe/rate-limit memory"},
                    "dedupe_window_seconds": {"type": "integer", "description": "Dedupe window for identical alerts (seconds)", "default": ALERT_ROUTING_DEFAULT_DEDUPE_WINDOW_SECONDS},
                    "rate_limit_window_seconds": {"type": "integer", "description": "Rate-limit window size (seconds)", "default": ALERT_ROUTING_DEFAULT_RATE_WINDOW_SECONDS},
                    "max_alerts_per_window": {"type": "integer", "description": "Maximum routed alerts per rate-limit window (0 disables)", "default": ALERT_ROUTING_DEFAULT_MAX_ALERTS_PER_WINDOW},
                    "webhook_retry_attempts": {"type": "integer", "description": "Webhook delivery attempts including first try", "default": ALERT_ROUTING_DEFAULT_WEBHOOK_RETRY_ATTEMPTS},
                    "webhook_backoff_ms": {"type": "integer", "description": "Initial webhook retry backoff in milliseconds", "default": ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MS},
                    "webhook_backoff_multiplier": {"type": "number", "description": "Exponential multiplier for webhook backoff", "default": ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MULTIPLIER},
                    "webhook_timeout_seconds": {"type": "integer", "description": "Per-attempt webhook request timeout", "default": ALERT_ROUTING_DEFAULT_WEBHOOK_TIMEOUT_SECONDS},
                    "webhook_retry_status_codes": {"type": "array", "description": "HTTP status codes treated as retryable", "items": {"type": "integer"}, "default": ALERT_ROUTING_DEFAULT_RETRY_STATUS_CODES}
                },
                "required": []
            }
        ),
        types.Tool(
            name="list-configs",
            description="List available training configurations",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get-config",
            description="Get a specific training configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Configuration name"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="get-training-info",
            description="Get training information including trigger word and test prompts",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Configuration name"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="upload-dataset",
            description="Upload images to create a new dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Name for the dataset"},
                    "images": {
                        "type": "array",
                        "description": "Array of images with their captions",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string", "description": "Image filename"},
                                "content": {"type": "string", "description": "Base64-encoded image content"},
                                "caption": {"type": "string", "description": "Caption for the image"}
                            },
                            "required": ["filename", "content", "caption"]
                        }
                    }
                },
                "required": ["dataset_name", "images"]
            }
        ),
        types.Tool(
            name="list-datasets",
            description="List available datasets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="list-comfyui-models",
            description="List local ComfyUI model files from configured model roots",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional filename/path substring filter"},
                    "limit": {"type": "integer", "description": "Maximum results to return (1-500)", "default": 50}
                },
                "required": []
            }
        ),
        types.Tool(
            name="start-training",
            description="Start a training job with a configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_name": {"type": "string", "description": "Name of the configuration to use"}
                },
                "required": ["config_name"]
            }
        ),
        types.Tool(
            name="get-training-status",
            description="Get the status of a training job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"}
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="stop-training",
            description="Stop a running training job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID to stop"}
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="list-training-jobs",
            description="List all training jobs",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="export-model",
            description="Export a trained model",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                    "format": {"type": "string", "description": "Export format", "enum": ["safetensors", "ckpt"], "default": "safetensors"}
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="list-exported-models",
            description="List exported models",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="download-model",
            description="Download a trained LoRA model as base64-encoded content",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Path to the model file (relative to outputs directory)"},
                    "include_metadata": {"type": "boolean", "description": "Include training metadata if available", "default": True}
                },
                "required": ["model_path"]
            }
        ),
        types.Tool(
            name="get-system-stats",
            description="Get AI Toolkit system statistics",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get-training-observability",
            description="Comprehensive log intelligence report (speed, quality, dataset, NLP, evaluation, and recommendations)",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                    "lines": {"type": "integer", "description": "Number of log lines to analyze", "default": 500},
                    "config_name": {"type": "string", "description": "Optional saved config name for dataset resolution"},
                    "dataset_path": {"type": "string", "description": "Optional explicit dataset path override"},
                    "include_dataset": {"type": "boolean", "description": "Include dataset diagnostics", "default": True},
                    "include_nlp": {"type": "boolean", "description": "Include NLP summaries on logs/captions", "default": True},
                    "include_raw_log_excerpt": {"type": "boolean", "description": "Include raw log excerpt in response", "default": False},
                    "excerpt_lines": {"type": "integer", "description": "Raw excerpt line count when include_raw_log_excerpt=true", "default": 120},
                    "max_dataset_files": {"type": "integer", "description": "Maximum image files scanned for dataset diagnostics", "default": 5000},
                    "include_evaluation": {"type": "boolean", "description": "Attach weighted run evaluation output", "default": True},
                    "include_next_experiment": {"type": "boolean", "description": "Attach highest-value next experiment recommendation", "default": True},
                    "pass_threshold": {"type": "number", "description": "Pass threshold used by evaluation rubric", "default": 70},
                    "include_baseline_comparison": {"type": "boolean", "description": "Compare this run against baseline runs", "default": False},
                    "baseline_job_ids": {"type": "array", "description": "Optional baseline job IDs for comparison", "items": {"type": "string"}},
                    "baseline_limit": {"type": "integer", "description": "When baseline_job_ids is omitted, compare against up to N recent runs", "default": 5}
                },
                "required": ["job_id"]
            }
        ),
        types.Tool(
            name="get-training-logs",
            description="Get logs for a training job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Training job ID"},
                    "lines": {"type": "integer", "description": "Number of log lines to retrieve", "default": 100}
                },
                "required": ["job_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, 
    arguments: dict | None
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests"""
    
    if name not in MCP_TOOLS:
        raise ValueError(f"Unknown tool: {name}")
        
    # Ensure arguments is not None
    if arguments is None:
        arguments = {}
        
    client = AIToolkitClient(AI_TOOLKIT_SERVER_URL)
    
    try:
        if name == "create-training-config":
            # Validate required parameters
            required_params = ["name", "model_name", "dataset_path"]
            missing_params = [p for p in required_params if not arguments.get(p)]
            if missing_params:
                return [types.TextContent(
                    type="text",
                    text=f"Error: Missing required parameters: {', '.join(missing_params)}"
                )]

            requested_model = arguments.get("model_name")
            resolved_model, resolved_source = resolve_model_reference(requested_model)
            requested_text = str(requested_model or "")
            resolved_text = str(resolved_model or "")
            is_flux_request = ("flux" in requested_text.lower() or "flex" in requested_text.lower() or
                               "flux" in resolved_text.lower() or "flex" in resolved_text.lower())
            resolved_path = Path(resolved_text).expanduser()

            if is_flux_request and resolved_path.exists():
                if resolved_path.is_file():
                    return [types.TextContent(
                        type="text",
                        text=(
                            "Error: Flux/Flex training cannot use a single checkpoint file path.\n"
                            f"Resolved model path is a file: {resolved_path}\n"
                            "Use a HuggingFace diffusers model ID (e.g., 'ostris/Flex.1-alpha') or "
                            "a local diffusers model directory that contains model_index.json and/or transformer/config.json."
                        )
                    )]
                if resolved_path.is_dir() and not is_diffusers_style_model_dir(resolved_path):
                    return [types.TextContent(
                        type="text",
                        text=(
                            "Error: Flux/Flex local model directory is missing diffusers config files.\n"
                            f"Resolved model path: {resolved_path}\n"
                            "Expected: model_index.json or transformer/config.json."
                        )
                    )]
            
            # Create training configuration
            config = create_lora_config(
                name=arguments.get("name"),
                model_name=resolved_model,
                dataset_path=arguments.get("dataset_path"),
                resolution=arguments.get("resolution", 512),
                batch_size=arguments.get("batch_size", 1),
                learning_rate=arguments.get("learning_rate", 2e-4),
                steps=arguments.get("steps", 1000),
                rank=arguments.get("rank", 16),
                alpha=arguments.get("alpha", 16),
                use_wandb=arguments.get("use_wandb", False),
                low_vram=arguments.get("low_vram", True),
                trigger_word=arguments.get("trigger_word"),
                test_prompts=arguments.get("test_prompts"),
                disable_sampling=arguments.get("disable_sampling", False)
            )
            
            # Save configuration
            config_path = save_config(arguments.get("name"), config)
            
            result = f"Created training configuration '{arguments.get('name')}':\n"
            result += f"- Saved to: {config_path}\n"
            if resolved_model != requested_model:
                result += f"- Model (requested): {requested_model}\n"
                result += f"- Model (resolved local path): {resolved_model}\n"
                if resolved_source:
                    result += f"- Resolved from: {resolved_source}\n"
            else:
                result += f"- Model: {resolved_model}\n"
            result += f"- Dataset: {arguments.get('dataset_path')}\n"
            result += f"- Steps: {arguments.get('steps', 1000)}\n"
            result += f"- LoRA rank: {arguments.get('rank', 16)}"
                
            return [types.TextContent(type="text", text=result)]

        elif name == "validate-training-config":
            check_dataset = bool(arguments.get("check_dataset", True))
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 2000

            config_source = None
            requested_config_name = arguments.get("config_name")
            config_payload: Optional[Dict[str, Any]] = None

            raw_config = arguments.get("config")
            if raw_config is not None:
                parsed = _parse_structured_config(raw_config)
                if not parsed:
                    return [types.TextContent(
                        type="text",
                        text=(
                            "Error: `config` must be a dict or a valid JSON/YAML object string."
                        )
                    )]
                config_payload = parsed
                config_source = "config_argument"
            elif requested_config_name:
                loaded = load_config(requested_config_name)
                if not loaded:
                    return [types.TextContent(
                        type="text",
                        text=f"Configuration '{requested_config_name}' not found."
                    )]
                config_payload = loaded
                config_source = "saved_config"
            elif arguments.get("model_name") and arguments.get("dataset_path"):
                preview_name = str(arguments.get("name") or "__validation_preview__")
                resolved_model, _ = resolve_model_reference(str(arguments.get("model_name")))
                try:
                    config_payload = create_lora_config(
                        name=preview_name,
                        model_name=resolved_model,
                        dataset_path=str(arguments.get("dataset_path")),
                        resolution=arguments.get("resolution", 512),
                        batch_size=arguments.get("batch_size", 1),
                        learning_rate=arguments.get("learning_rate", 2e-4),
                        steps=arguments.get("steps", 1000),
                        rank=arguments.get("rank", 16),
                        alpha=arguments.get("alpha", 16),
                        use_wandb=bool(arguments.get("use_wandb", False)),
                        low_vram=bool(arguments.get("low_vram", True)),
                        trigger_word=arguments.get("trigger_word"),
                        test_prompts=arguments.get("test_prompts"),
                        disable_sampling=bool(arguments.get("disable_sampling", False)),
                    )
                except Exception as exc:
                    return [types.TextContent(
                        type="text",
                        text=f"Error: Could not build preview config for validation: {exc}"
                    )]
                config_source = "draft_parameters"
            else:
                return [types.TextContent(
                    type="text",
                    text=(
                        "Error: Provide one of: `config_name`, `config`, "
                        "or draft parameters including `model_name` and `dataset_path`."
                    )
                )]

            validation = validate_training_config_payload(
                config_payload,
                check_dataset=check_dataset,
                max_dataset_files=max_dataset_files,
            )
            validation["config_context"] = {
                "config_source": config_source,
                "requested_config_name": requested_config_name,
                "check_dataset": check_dataset,
                "max_dataset_files": max_dataset_files,
            }
            return [types.TextContent(type="text", text=json.dumps(validation, indent=2))]

        elif name == "recommend-training-plan":
            include_dataset_scan = bool(arguments.get("include_dataset_scan", True))
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 2000

            goal = str(arguments.get("goal", "") or "").strip()
            base_model = str(arguments.get("base_model") or arguments.get("model_name") or "").strip()
            dataset_path_value = arguments.get("dataset_path")
            config_name = arguments.get("config_name")
            config_source = None
            config_payload: Optional[Dict[str, Any]] = None

            if config_name:
                loaded = load_config(config_name)
                if not loaded:
                    return [types.TextContent(
                        type="text",
                        text=f"Configuration '{config_name}' not found."
                    )]
                config_payload = loaded
                config_source = "saved_config"
                if not base_model:
                    cfg = loaded.get("config", loaded)
                    process = cfg.get("process", []) if isinstance(cfg, dict) else []
                    if isinstance(process, list) and process and isinstance(process[0], dict):
                        model_obj = process[0].get("model", {})
                        if isinstance(model_obj, dict):
                            base_model = str(model_obj.get("name_or_path") or "").strip()
                if not dataset_path_value:
                    inferred_dataset = _extract_dataset_path_from_config(loaded)
                    if inferred_dataset:
                        dataset_path_value = inferred_dataset

            if not base_model:
                if "flux" in goal.lower() or "flex" in goal.lower():
                    base_model = "ostris/Flex.1-alpha"
                else:
                    base_model = "runwayml/stable-diffusion-v1-5"
                if not config_source:
                    config_source = "heuristic_default"

            time_budget_minutes = _to_float(arguments.get("time_budget_minutes"))
            target_vram_gb = _to_float(arguments.get("target_vram_gb"))

            system_snapshot = await client.get_system_stats()
            if target_vram_gb is None:
                total_mb = _to_float(system_snapshot.get("gpu_memory_total"))
                if total_mb and total_mb > 0:
                    target_vram_gb = round(total_mb / 1024.0, 2)

            dataset_analysis = None
            dataset_source = None
            if include_dataset_scan and dataset_path_value:
                dataset_source = "argument"
                if config_source == "saved_config" and arguments.get("dataset_path") is None:
                    dataset_source = "config"
                dataset_analysis, _ = analyze_dataset_folder(Path(str(dataset_path_value)), max_files=max_dataset_files)
            elif include_dataset_scan and not dataset_path_value:
                dataset_analysis = {
                    "dataset_path": None,
                    "exists": False,
                    "error": "dataset_path_unresolved"
                }

            plan = recommend_training_plan_payload(
                goal=goal,
                base_model=base_model,
                dataset_analysis=dataset_analysis,
                target_vram_gb=target_vram_gb,
                time_budget_minutes=time_budget_minutes,
            )

            result = {
                "generated_at": _utc_now_iso(),
                "inputs": {
                    "goal": goal,
                    "base_model": base_model,
                    "dataset_path": str(dataset_path_value) if dataset_path_value else None,
                    "time_budget_minutes": time_budget_minutes,
                    "target_vram_gb": target_vram_gb,
                    "include_dataset_scan": include_dataset_scan,
                },
                "context": {
                    "config_name": config_name,
                    "config_source": config_source,
                    "dataset_source": dataset_source,
                    "max_dataset_files": max_dataset_files,
                },
                "dataset_analysis": dataset_analysis if dataset_analysis is not None else {"skipped": True},
                "system_snapshot": system_snapshot,
                "plan": plan,
                "next_actions": [
                    "Run `validate-training-config` on the chosen profile before launch.",
                    "Create/update a config with `create-training-config` using the recommended plan.",
                    "Start job and monitor via `get-training-observability`.",
                ],
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "run-training-with-guardrails":
            config_name = str(arguments.get("config_name") or "").strip()
            if not config_name:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: config_name"
                )]

            config_payload = load_config(config_name)
            if not config_payload:
                return [types.TextContent(
                    type="text",
                    text=f"Configuration '{config_name}' not found."
                )]

            check_dataset = bool(arguments.get("check_dataset", True))
            block_on_validation_fail = bool(arguments.get("block_on_validation_fail", True))
            stop_on_nan = bool(arguments.get("stop_on_nan", True))
            stop_on_repeated_oom = bool(arguments.get("stop_on_repeated_oom", True))
            stop_on_stalled_progress = bool(arguments.get("stop_on_stalled_progress", True))
            stop_on_throughput_collapse = bool(arguments.get("stop_on_throughput_collapse", True))

            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 2000
            try:
                monitor_seconds = max(0, min(int(arguments.get("monitor_seconds", 120)), 3600))
            except (TypeError, ValueError):
                monitor_seconds = 120
            try:
                poll_interval_seconds = max(2, min(int(arguments.get("poll_interval_seconds", 10)), 300))
            except (TypeError, ValueError):
                poll_interval_seconds = 10
            try:
                oom_threshold = max(1, min(int(arguments.get("oom_threshold", 3)), 100))
            except (TypeError, ValueError):
                oom_threshold = 3
            try:
                stall_window_seconds = max(10, min(int(arguments.get("stall_window_seconds", 120)), 3600))
            except (TypeError, ValueError):
                stall_window_seconds = 120
            try:
                min_step_delta = max(1, min(int(arguments.get("min_step_delta", 1)), 1000))
            except (TypeError, ValueError):
                min_step_delta = 1
            try:
                throughput_floor_iter_per_sec = max(0.0, float(arguments.get("throughput_floor_iter_per_sec", 0.02)))
            except (TypeError, ValueError):
                throughput_floor_iter_per_sec = 0.02
            try:
                low_speed_consecutive_polls = max(1, min(int(arguments.get("low_speed_consecutive_polls", 2)), 20))
            except (TypeError, ValueError):
                low_speed_consecutive_polls = 2
            try:
                log_lines = max(100, min(int(arguments.get("log_lines", 400)), 20000))
            except (TypeError, ValueError):
                log_lines = 400

            guardrails = {
                "stop_on_nan": stop_on_nan,
                "stop_on_repeated_oom": stop_on_repeated_oom,
                "oom_threshold": oom_threshold,
                "stop_on_stalled_progress": stop_on_stalled_progress,
                "stall_window_seconds": stall_window_seconds,
                "min_step_delta": min_step_delta,
                "stop_on_throughput_collapse": stop_on_throughput_collapse,
                "throughput_floor_iter_per_sec": throughput_floor_iter_per_sec,
                "low_speed_consecutive_polls": low_speed_consecutive_polls,
            }

            validation = validate_training_config_payload(
                config_payload,
                check_dataset=check_dataset,
                max_dataset_files=max_dataset_files,
            )
            if block_on_validation_fail and not validation.get("valid", False):
                return [types.TextContent(type="text", text=json.dumps({
                    "started": False,
                    "status": "blocked_by_validation",
                    "config_name": config_name,
                    "guardrails": guardrails,
                    "validation": validation,
                }, indent=2))]

            config_path = CONFIG_DIR / f"{config_name}.yaml"
            start_resp = await client.start_training(str(config_path))
            if not start_resp.get("success"):
                return [types.TextContent(type="text", text=json.dumps({
                    "started": False,
                    "status": "start_failed",
                    "config_name": config_name,
                    "guardrails": guardrails,
                    "validation": validation,
                    "error": start_resp.get("error", "Unknown error"),
                }, indent=2))]

            job_id = start_resp.get("job_id")
            monitor_snapshots: List[Dict[str, Any]] = []
            triggered_guardrails: List[Dict[str, Any]] = []
            low_speed_counter = 0
            action = "started"
            stop_result = None
            loop = asyncio.get_running_loop()
            started_at = loop.time()
            last_progress_at = started_at
            last_progress_step = 0

            if monitor_seconds > 0:
                deadline = started_at + monitor_seconds
                while loop.time() < deadline:
                    status_snapshot = await client.get_training_status(job_id)
                    logs_response = await client.get_training_logs(job_id, log_lines)
                    log_text = logs_response.get("log", "") if logs_response.get("success") else ""
                    speed_analysis = analyze_speed_from_logs(log_text, speed_string=status_snapshot.get("speed_string", ""))
                    quality_analysis = analyze_quality_from_logs(log_text)
                    error_analysis = analyze_error_signals(log_text)

                    current_step = int(status_snapshot.get("current_step", 0) or 0)
                    now_ts = loop.time()
                    if current_step > last_progress_step:
                        last_progress_step = current_step
                        last_progress_at = now_ts
                    seconds_since_progress = now_ts - last_progress_at

                    triggers, low_speed_counter = detect_guardrail_triggers(
                        quality_analysis=quality_analysis,
                        error_analysis=error_analysis,
                        speed_analysis=speed_analysis,
                        current_step=current_step,
                        last_progress_step=last_progress_step,
                        seconds_since_progress=seconds_since_progress,
                        consecutive_low_speed=low_speed_counter,
                        guardrails=guardrails,
                    )

                    monitor_snapshots.append({
                        "elapsed_seconds": round(now_ts - started_at, 1),
                        "status": status_snapshot.get("status"),
                        "current_step": current_step,
                        "progress_pct": status_snapshot.get("progress"),
                        "iter_per_sec_latest": (speed_analysis.get("iter_per_sec") or {}).get("latest"),
                        "quality_status": quality_analysis.get("status"),
                        "error_severity": error_analysis.get("severity"),
                        "triggers": [t.get("code") for t in triggers],
                    })

                    status_lower = str(status_snapshot.get("status", "")).lower()
                    if status_lower in {"completed", "failed", "error", "stopped"}:
                        action = f"job_{status_lower}"
                        break

                    if triggers:
                        triggered_guardrails.extend(triggers)
                        stop_result = await client.stop_training(job_id)
                        action = "stopped_by_guardrail" if stop_result.get("success") else "guardrail_triggered_stop_failed"
                        break

                    await asyncio.sleep(poll_interval_seconds)
                else:
                    action = "monitor_window_complete"

            final_status = await client.get_training_status(job_id)
            result = {
                "started": True,
                "status": action,
                "config_name": config_name,
                "job_id": job_id,
                "validation": validation,
                "guardrails": guardrails,
                "monitor": {
                    "monitor_seconds": monitor_seconds,
                    "poll_interval_seconds": poll_interval_seconds,
                    "snapshot_count": len(monitor_snapshots),
                    "snapshots": monitor_snapshots[-50:],
                },
                "triggered_guardrails": triggered_guardrails,
                "stop_result": stop_result,
                "final_status": final_status,
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "watch-training-timeseries":
            job_id = str(arguments.get("job_id") or "").strip()
            if not job_id:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: job_id"
                )]

            try:
                duration_seconds = max(0, min(int(arguments.get("duration_seconds", 120)), 7200))
            except (TypeError, ValueError):
                duration_seconds = 120
            try:
                poll_interval_seconds = max(2, min(int(arguments.get("poll_interval_seconds", 10)), 300))
            except (TypeError, ValueError):
                poll_interval_seconds = 10
            try:
                log_lines = max(100, min(int(arguments.get("log_lines", 400)), 20000))
            except (TypeError, ValueError):
                log_lines = 400
            try:
                excerpt_lines = max(10, min(int(arguments.get("excerpt_lines", 80)), 1000))
            except (TypeError, ValueError):
                excerpt_lines = 80
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 2000

            include_system = bool(arguments.get("include_system", True))
            include_alerts = bool(arguments.get("include_alerts", True))
            include_dataset = bool(arguments.get("include_dataset", False))
            include_nlp = bool(arguments.get("include_nlp", False))
            stop_on_terminal_status = bool(arguments.get("stop_on_terminal_status", True))
            include_raw_log_excerpt = bool(arguments.get("include_raw_log_excerpt", False))

            status_initial = await client.get_training_status(job_id)
            if not status_initial.get("found"):
                return [types.TextContent(type="text", text=f"Training job {job_id} not found.")]

            requested_config_name = arguments.get("config_name")
            config_source = None
            config_payload: Optional[Dict[str, Any]] = None
            resolved_config_name = requested_config_name

            if requested_config_name:
                config_payload = load_config(requested_config_name)
                if config_payload:
                    config_source = "saved_config"

            job_payload = await client.get_training_job(job_id)
            if config_payload is None and job_payload.get("found"):
                job_data = job_payload.get("job", {})
                parsed = _parse_structured_config(job_data.get("job_config"))
                if parsed:
                    config_payload = parsed
                    config_source = "job_payload"
                if not resolved_config_name:
                    resolved_config_name = job_data.get("name")

            if config_payload is None and resolved_config_name:
                from_disk = load_config(resolved_config_name)
                if from_disk:
                    config_payload = from_disk
                    config_source = "saved_config_by_job_name"

            dataset_source = None
            dataset_path_value = arguments.get("dataset_path")
            if dataset_path_value:
                dataset_source = "argument"
            else:
                inferred_path = _extract_dataset_path_from_config(config_payload or {})
                if inferred_path:
                    dataset_path_value = inferred_path
                    dataset_source = "config"

            dataset_analysis = None
            caption_texts: List[str] = []
            if include_dataset or include_nlp:
                if dataset_path_value:
                    dataset_analysis, caption_texts = analyze_dataset_folder(
                        Path(str(dataset_path_value)),
                        max_files=max_dataset_files
                    )
                else:
                    dataset_analysis = {
                        "dataset_path": None,
                        "exists": False,
                        "error": "dataset_path_unresolved"
                    }

            loop = asyncio.get_running_loop()
            started_at = loop.time()
            deadline = started_at + duration_seconds
            points: List[Dict[str, Any]] = []
            terminal_statuses = {"completed", "failed", "error", "stopped"}

            while True:
                now_ts = loop.time()
                status_snapshot = await client.get_training_status(job_id)
                logs_response = await client.get_training_logs(job_id, log_lines)
                log_text = logs_response.get("log", "") if logs_response.get("success") else ""
                log_tail_lines = log_text.splitlines()

                speed_analysis = analyze_speed_from_logs(log_text, speed_string=status_snapshot.get("speed_string", ""))
                quality_analysis = analyze_quality_from_logs(log_text)
                error_analysis = analyze_error_signals(log_text)

                nlp_analysis = None
                if include_nlp:
                    nlp_analysis = analyze_nlp_signals(log_text, caption_texts)

                alerts: List[Dict[str, Any]] = []
                if include_alerts:
                    alerts = build_observability_alerts(
                        speed_analysis=speed_analysis,
                        quality_analysis=quality_analysis,
                        error_analysis=error_analysis,
                        dataset_analysis=dataset_analysis if include_dataset else None,
                        nlp_analysis=nlp_analysis if include_nlp else None,
                    )
                health = overall_health_from_alerts(alerts)

                system_snapshot = await client.get_system_stats() if include_system else None
                point = {
                    "timestamp": _utc_now_iso(),
                    "elapsed_seconds": round(now_ts - started_at, 2),
                    "status": status_snapshot.get("status", "unknown"),
                    "current_step": status_snapshot.get("current_step"),
                    "total_steps": status_snapshot.get("total_steps"),
                    "progress_pct": status_snapshot.get("progress"),
                    "iter_per_sec": (speed_analysis.get("iter_per_sec") or {}).get("last"),
                    "speed_status": speed_analysis.get("status"),
                    "loss_last": (quality_analysis.get("loss") or {}).get("last"),
                    "quality_status": quality_analysis.get("status"),
                    "error_severity": error_analysis.get("severity"),
                    "error_mentions": error_analysis.get("total_mentions"),
                    "overall_health": health,
                    "alert_count": len(alerts),
                    "alert_max_severity": _max_alert_severity(alerts),
                }
                if include_system and system_snapshot:
                    point.update({
                        "gpu_utilization_pct": system_snapshot.get("gpu_utilization"),
                        "gpu_memory_used_mb": system_snapshot.get("gpu_memory_used"),
                        "gpu_memory_total_mb": system_snapshot.get("gpu_memory_total"),
                        "cpu_percent": system_snapshot.get("cpu_percent"),
                        "ram_used_gb": system_snapshot.get("ram_used"),
                        "ram_total_gb": system_snapshot.get("ram_total"),
                    })
                if include_alerts:
                    point["alerts"] = alerts
                if include_nlp and nlp_analysis is not None:
                    point["nlp"] = {
                        "caption_token_diversity": nlp_analysis.get("caption_token_diversity"),
                        "style_flags": nlp_analysis.get("style_flags"),
                    }
                if include_raw_log_excerpt:
                    point["raw_log_excerpt"] = "\n".join(log_tail_lines[-excerpt_lines:])

                points.append(point)

                status_lower = str(point.get("status", "")).lower()
                if stop_on_terminal_status and status_lower in terminal_statuses:
                    break
                if now_ts >= deadline:
                    break
                if duration_seconds == 0:
                    break

                await asyncio.sleep(poll_interval_seconds)

            summary = summarize_timeseries_points(points)
            nlp_summary = None
            if include_nlp and points:
                style_counter = Counter()
                diversity_values: List[float] = []
                for p in points:
                    nlp_point = p.get("nlp", {}) if isinstance(p.get("nlp"), dict) else {}
                    for flag in nlp_point.get("style_flags", []) or []:
                        style_counter[str(flag)] += 1
                    div_val = _to_float(nlp_point.get("caption_token_diversity"))
                    if div_val is not None:
                        diversity_values.append(div_val)
                nlp_summary = {
                    "style_flag_counts": dict(style_counter),
                    "caption_token_diversity_mean": round(statistics.fmean(diversity_values), 6) if diversity_values else None,
                }

            final_status = await client.get_training_status(job_id)
            result = {
                "job_id": job_id,
                "generated_at": _utc_now_iso(),
                "window": {
                    "duration_seconds_requested": duration_seconds,
                    "poll_interval_seconds": poll_interval_seconds,
                    "points_collected": len(points),
                    "elapsed_seconds_actual": round(points[-1]["elapsed_seconds"], 2) if points else 0.0,
                },
                "config_context": {
                    "requested_config_name": requested_config_name,
                    "resolved_config_name": resolved_config_name,
                    "config_source": config_source,
                },
                "dataset_context": {
                    "dataset_path": str(dataset_path_value) if dataset_path_value else None,
                    "dataset_source": dataset_source,
                    "dataset_analysis": dataset_analysis if include_dataset else {"skipped": True},
                },
                "summary": summary,
                "nlp_summary": nlp_summary if include_nlp else {"skipped": True},
                "points": points,
                "final_status": final_status,
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "compare-training-runs":
            requested_job_ids = arguments.get("job_ids", [])
            if requested_job_ids is None:
                requested_job_ids = []
            if not isinstance(requested_job_ids, list):
                return [types.TextContent(
                    type="text",
                    text="Error: job_ids must be an array of strings when provided."
                )]

            try:
                limit = max(1, min(int(arguments.get("limit", 5)), 50))
            except (TypeError, ValueError):
                limit = 5
            try:
                log_lines = max(100, min(int(arguments.get("log_lines", 800)), 20000))
            except (TypeError, ValueError):
                log_lines = 800
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 2000

            include_dataset = bool(arguments.get("include_dataset", True))
            include_nlp = bool(arguments.get("include_nlp", False))
            only_terminal_runs = bool(arguments.get("only_terminal_runs", False))
            terminal_statuses = {"completed", "failed", "error", "stopped"}

            comparison_job_ids: List[str] = []
            if requested_job_ids:
                comparison_job_ids = [str(job_id).strip() for job_id in requested_job_ids if str(job_id).strip()]
            else:
                jobs_resp = await client.list_training_jobs()
                if not jobs_resp.get("success"):
                    return [types.TextContent(
                        type="text",
                        text=f"Failed to list jobs for comparison: {jobs_resp.get('error', 'Unknown error')}"
                    )]
                jobs = jobs_resp.get("jobs", [])
                for job in jobs:
                    job_id = str(job.get("id") or "").strip()
                    if not job_id:
                        continue
                    if only_terminal_runs and str(job.get("status", "")).lower() not in terminal_statuses:
                        continue
                    comparison_job_ids.append(job_id)
                    if len(comparison_job_ids) >= limit:
                        break

            if not comparison_job_ids:
                return [types.TextContent(
                    type="text",
                    text="No training jobs available for comparison."
                )]

            dataset_cache: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
            run_reports: List[Dict[str, Any]] = []
            skipped_runs: List[Dict[str, Any]] = []

            for job_id in comparison_job_ids:
                status_snapshot = await client.get_training_status(job_id)
                if not status_snapshot.get("found"):
                    skipped_runs.append({"job_id": job_id, "reason": "job_not_found"})
                    continue

                status_lower = str(status_snapshot.get("status", "")).lower()
                if only_terminal_runs and status_lower not in terminal_statuses:
                    skipped_runs.append({"job_id": job_id, "reason": "non_terminal_status", "status": status_snapshot.get("status")})
                    continue

                report_response = await collect_job_run_report(
                    client,
                    job_id,
                    log_lines=log_lines,
                    include_dataset=include_dataset,
                    include_nlp=include_nlp,
                    max_dataset_files=max_dataset_files,
                    dataset_cache=dataset_cache,
                )
                if not report_response.get("found"):
                    skipped_runs.append({"job_id": job_id, "reason": report_response.get("error", "collection_failed")})
                    continue
                report = report_response.get("run_report", {})
                if isinstance(report, dict):
                    report.pop("config_payload", None)
                run_reports.append(report)

            ranking = rank_run_comparisons(run_reports)
            result = {
                "generated_at": _utc_now_iso(),
                "inputs": {
                    "job_ids": comparison_job_ids,
                    "requested_job_ids": requested_job_ids,
                    "log_lines": log_lines,
                    "include_dataset": include_dataset,
                    "include_nlp": include_nlp,
                    "only_terminal_runs": only_terminal_runs,
                },
                "compared_count": len(run_reports),
                "skipped_runs": skipped_runs,
                "ranking": ranking,
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "evaluate-lora":
            job_id = str(arguments.get("job_id") or "").strip()
            if not job_id:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: job_id"
                )]

            try:
                log_lines = max(100, min(int(arguments.get("log_lines", 1000)), 20000))
            except (TypeError, ValueError):
                log_lines = 1000
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 2000
            pass_threshold = _to_float(arguments.get("pass_threshold"))
            if pass_threshold is None:
                pass_threshold = 70.0

            include_dataset = bool(arguments.get("include_dataset", True))
            include_nlp = bool(arguments.get("include_nlp", True))
            rubric_weights = arguments.get("rubric_weights")
            if rubric_weights is not None and not isinstance(rubric_weights, dict):
                return [types.TextContent(
                    type="text",
                    text="Error: rubric_weights must be an object when provided."
                )]

            report_response = await collect_job_run_report(
                client,
                job_id,
                log_lines=log_lines,
                include_dataset=include_dataset,
                include_nlp=include_nlp,
                max_dataset_files=max_dataset_files,
                dataset_cache={},
            )
            if not report_response.get("found"):
                return [types.TextContent(
                    type="text",
                    text=f"Failed to collect run diagnostics for {job_id}: {report_response.get('error', 'Unknown error')}"
                )]

            run_report = report_response.get("run_report", {})
            evaluation = evaluate_run_report(
                run_report,
                rubric_weights=rubric_weights,
                pass_threshold=float(pass_threshold),
            )

            output_report = dict(run_report)
            output_report.pop("config_payload", None)

            result = {
                "generated_at": _utc_now_iso(),
                "job_id": job_id,
                "inputs": {
                    "log_lines": log_lines,
                    "include_dataset": include_dataset,
                    "include_nlp": include_nlp,
                    "pass_threshold": pass_threshold,
                    "rubric_weights": rubric_weights or {},
                },
                "evaluation": evaluation,
                "run_report": output_report,
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "suggest-next-experiment":
            requested_job_id = str(arguments.get("job_id") or "").strip()
            requested_job_ids = arguments.get("job_ids", [])
            if requested_job_ids is None:
                requested_job_ids = []
            if not isinstance(requested_job_ids, list):
                return [types.TextContent(
                    type="text",
                    text="Error: job_ids must be an array of strings when provided."
                )]

            try:
                log_lines = max(100, min(int(arguments.get("log_lines", 1000)), 20000))
            except (TypeError, ValueError):
                log_lines = 1000
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 2000
            pass_threshold = _to_float(arguments.get("pass_threshold"))
            if pass_threshold is None:
                pass_threshold = 70.0

            include_dataset = bool(arguments.get("include_dataset", True))
            include_nlp = bool(arguments.get("include_nlp", True))

            candidate_ids: List[str] = []
            if requested_job_id:
                candidate_ids.append(requested_job_id)
            candidate_ids.extend([str(x).strip() for x in requested_job_ids if str(x).strip()])
            candidate_ids = list(dict.fromkeys(candidate_ids))

            if not candidate_ids:
                jobs_resp = await client.list_training_jobs()
                if not jobs_resp.get("success"):
                    return [types.TextContent(
                        type="text",
                        text=f"Failed to list jobs for recommendation: {jobs_resp.get('error', 'Unknown error')}"
                    )]
                candidate_ids = [str(job.get("id") or "").strip() for job in jobs_resp.get("jobs", []) if str(job.get("id") or "").strip()][:5]

            if not candidate_ids:
                return [types.TextContent(
                    type="text",
                    text="No candidate jobs available for next-experiment suggestion."
                )]

            dataset_cache: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
            evaluated_candidates: List[Dict[str, Any]] = []
            skipped_candidates: List[Dict[str, Any]] = []

            for job_id in candidate_ids:
                report_response = await collect_job_run_report(
                    client,
                    job_id,
                    log_lines=log_lines,
                    include_dataset=include_dataset,
                    include_nlp=include_nlp,
                    max_dataset_files=max_dataset_files,
                    dataset_cache=dataset_cache,
                )
                if not report_response.get("found"):
                    skipped_candidates.append({"job_id": job_id, "reason": report_response.get("error", "collection_failed")})
                    continue
                run_report = report_response.get("run_report", {})
                evaluation = evaluate_run_report(run_report, pass_threshold=float(pass_threshold))
                evaluated_candidates.append({
                    "job_id": job_id,
                    "evaluation": evaluation,
                    "run_report": run_report,
                })

            if not evaluated_candidates:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "generated_at": _utc_now_iso(),
                        "status": "no_evaluable_candidates",
                        "requested_job_ids": candidate_ids,
                        "skipped_candidates": skipped_candidates,
                    }, indent=2)
                )]

            evaluated_candidates.sort(
                key=lambda item: item.get("evaluation", {}).get("overall_score", 0.0),
                reverse=True
            )
            selected = evaluated_candidates[0]
            selected_report = selected.get("run_report", {})
            selected_eval = selected.get("evaluation", {})
            recommendation = suggest_next_experiment_from_run(
                selected_report,
                selected_eval,
                selected_report.get("config_payload"),
            )

            output_report = dict(selected_report)
            output_report.pop("config_payload", None)

            result = {
                "generated_at": _utc_now_iso(),
                "inputs": {
                    "candidate_job_ids": candidate_ids,
                    "log_lines": log_lines,
                    "include_dataset": include_dataset,
                    "include_nlp": include_nlp,
                    "pass_threshold": pass_threshold,
                },
                "selected_job_id": selected.get("job_id"),
                "selected_evaluation": selected_eval,
                "selected_run_report": output_report,
                "recommendation": recommendation,
                "candidate_scores": [
                    {
                        "job_id": item.get("job_id"),
                        "overall_score": item.get("evaluation", {}).get("overall_score"),
                        "grade": item.get("evaluation", {}).get("grade"),
                        "passed": item.get("evaluation", {}).get("passed"),
                    }
                    for item in evaluated_candidates
                ],
                "skipped_candidates": skipped_candidates,
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "auto-improve-dataset":
            dataset_path_value = arguments.get("dataset_path")
            config_name = arguments.get("config_name")
            config_source = None
            if not dataset_path_value and config_name:
                loaded_config = load_config(str(config_name))
                if loaded_config:
                    dataset_path_value = _extract_dataset_path_from_config(loaded_config)
                    config_source = "config"

            if not dataset_path_value:
                return [types.TextContent(
                    type="text",
                    text="Error: Provide `dataset_path` or `config_name` that resolves to a dataset path."
                )]

            dataset_path = Path(str(dataset_path_value))
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 5000)), 50000))
            except (TypeError, ValueError):
                max_dataset_files = 5000

            apply_changes = bool(arguments.get("apply", False))
            trigger_word = arguments.get("trigger_word")
            fill_missing_captions = bool(arguments.get("fill_missing_captions", True))
            fill_empty_captions = bool(arguments.get("fill_empty_captions", True))
            quarantine_duplicates = bool(arguments.get("quarantine_duplicates", False))
            quarantine_corrupt = bool(arguments.get("quarantine_corrupt", True))

            dataset_analysis_before, _ = analyze_dataset_folder(dataset_path, max_files=max_dataset_files)
            inspection = inspect_dataset_files(dataset_path, max_files=max_dataset_files)
            if not inspection.get("exists"):
                return [types.TextContent(
                    type="text",
                    text=f"Dataset path not found or not a directory: {dataset_path}"
                )]

            plan_actions: List[Dict[str, Any]] = []
            if inspection.get("missing_caption_files"):
                plan_actions.append({
                    "action": "create_missing_captions",
                    "count": len(inspection.get("missing_caption_files", [])),
                    "enabled": fill_missing_captions,
                })
            if inspection.get("empty_caption_files"):
                plan_actions.append({
                    "action": "fill_empty_captions",
                    "count": len(inspection.get("empty_caption_files", [])),
                    "enabled": fill_empty_captions,
                })
            if inspection.get("duplicate_files"):
                plan_actions.append({
                    "action": "quarantine_duplicates",
                    "count": len(inspection.get("duplicate_files", [])),
                    "enabled": quarantine_duplicates,
                })
            if inspection.get("corrupt_files"):
                plan_actions.append({
                    "action": "quarantine_corrupt",
                    "count": len(inspection.get("corrupt_files", [])),
                    "enabled": quarantine_corrupt,
                })

            apply_result = None
            dataset_analysis_after = None
            if apply_changes:
                apply_result = apply_dataset_improvements(
                    dataset_path,
                    inspection,
                    trigger_word=trigger_word,
                    fill_missing_captions=fill_missing_captions,
                    fill_empty_captions=fill_empty_captions,
                    quarantine_duplicates=quarantine_duplicates,
                    quarantine_corrupt=quarantine_corrupt,
                )
                dataset_analysis_after, _ = analyze_dataset_folder(dataset_path, max_files=max_dataset_files)

            result = {
                "generated_at": _utc_now_iso(),
                "dataset_path": str(dataset_path),
                "config_name": config_name,
                "config_source": config_source,
                "applied": apply_changes,
                "options": {
                    "fill_missing_captions": fill_missing_captions,
                    "fill_empty_captions": fill_empty_captions,
                    "quarantine_duplicates": quarantine_duplicates,
                    "quarantine_corrupt": quarantine_corrupt,
                    "trigger_word": trigger_word,
                },
                "analysis_before": dataset_analysis_before,
                "inspection": inspection,
                "plan_actions": plan_actions,
                "apply_result": apply_result if apply_changes else {"skipped": True},
                "analysis_after": dataset_analysis_after if apply_changes else {"skipped": True},
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "export-observability-report":
            job_id = str(arguments.get("job_id") or "").strip()
            if not job_id:
                return [types.TextContent(type="text", text="Error: Missing required parameter: job_id")]

            try:
                lines = max(100, min(int(arguments.get("lines", 800)), 20000))
            except (TypeError, ValueError):
                lines = 800
            try:
                max_dataset_files = max(100, min(int(arguments.get("max_dataset_files", 2000)), 50000))
            except (TypeError, ValueError):
                max_dataset_files = 2000
            include_dataset = bool(arguments.get("include_dataset", True))
            include_nlp = bool(arguments.get("include_nlp", True))
            include_evaluation = bool(arguments.get("include_evaluation", True))
            include_raw_log_excerpt = bool(arguments.get("include_raw_log_excerpt", False))
            config_name = arguments.get("config_name")
            dataset_path_override = arguments.get("dataset_path")
            filename_prefix = str(arguments.get("filename_prefix") or "observability").strip() or "observability"
            output_dir_str = arguments.get("output_dir")
            output_dir = Path(str(output_dir_str)) if output_dir_str else REPORTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            snapshot = await collect_observability_snapshot(
                client,
                job_id=job_id,
                lines=lines,
                config_name=config_name,
                dataset_path_override=dataset_path_override,
                include_dataset=include_dataset,
                include_nlp=include_nlp,
                include_raw_log_excerpt=include_raw_log_excerpt,
                excerpt_lines=120,
                max_dataset_files=max_dataset_files,
            )
            if not snapshot.get("success"):
                return [types.TextContent(
                    type="text",
                    text=f"Failed to collect observability snapshot: {snapshot.get('error', 'Unknown error')}"
                )]

            evaluation = None
            if include_evaluation:
                run_response = await collect_job_run_report(
                    client,
                    job_id,
                    log_lines=lines,
                    include_dataset=include_dataset,
                    include_nlp=include_nlp,
                    max_dataset_files=max_dataset_files,
                    dataset_cache={},
                )
                if run_response.get("found"):
                    run_report = run_response.get("run_report", {})
                    evaluation = evaluate_run_report(run_report, pass_threshold=70.0)
                    run_report.pop("config_payload", None)
                    snapshot["run_report"] = run_report
                else:
                    evaluation = {"error": run_response.get("error", "evaluation_collection_failed")}

            report_payload = {
                "generated_at": _utc_now_iso(),
                "job_id": job_id,
                "snapshot": snapshot,
                "evaluation": evaluation if include_evaluation else {"skipped": True},
            }

            safe_job_id = re.sub(r"[^A-Za-z0-9._-]+", "_", job_id)
            timestamp = _utc_now().strftime("%Y%m%d-%H%M%S")
            basename = f"{filename_prefix}-{safe_job_id}-{timestamp}"
            json_path = output_dir / f"{basename}.json"
            md_path = output_dir / f"{basename}.md"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=2)

            alerts = snapshot.get("alerts", [])
            speed_mean = ((snapshot.get("speed_analysis") or {}).get("iter_per_sec") or {}).get("mean")
            convergence = (snapshot.get("quality_analysis") or {}).get("convergence_score")
            overall_health = snapshot.get("overall_health")
            eval_score = (evaluation or {}).get("overall_score") if isinstance(evaluation, dict) else None

            md_lines = [
                f"# Observability Report: {job_id}",
                "",
                f"- Generated: {report_payload['generated_at']}",
                f"- Overall health: {overall_health}",
                f"- Speed mean (iter/s): {speed_mean}",
                f"- Convergence score: {convergence}",
                f"- Alert count: {len(alerts)}",
            ]
            if eval_score is not None:
                md_lines.append(f"- Evaluation score: {eval_score}")
                md_lines.append(f"- Evaluation grade: {(evaluation or {}).get('grade')}")
            md_lines.extend(["", "## Alerts", ""])
            if alerts:
                for alert in alerts:
                    md_lines.append(
                        f"- [{alert.get('severity', 'info')}] {alert.get('category', 'general')}: {alert.get('message')}"
                    )
            else:
                md_lines.append("- No alerts")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines) + "\n")

            result = {
                "success": True,
                "job_id": job_id,
                "json_report_path": str(json_path),
                "markdown_report_path": str(md_path),
                "output_dir": str(output_dir),
                "overall_health": overall_health,
                "alert_count": len(alerts),
                "evaluation_score": eval_score,
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "alert-routing":
            sink = str(arguments.get("sink") or "file").strip().lower()
            destination = arguments.get("destination")
            include_context = bool(arguments.get("include_context", True))
            min_severity = str(arguments.get("min_severity") or "warning").strip().lower()
            if min_severity not in {"info", "warning", "critical"}:
                min_severity = "warning"
            threshold = SEVERITY_ORDER.get(min_severity, 2)
            try:
                dedupe_window_seconds = max(
                    0,
                    min(
                        int(arguments.get("dedupe_window_seconds", ALERT_ROUTING_DEFAULT_DEDUPE_WINDOW_SECONDS)),
                        7 * 24 * 3600,
                    ),
                )
            except (TypeError, ValueError):
                dedupe_window_seconds = ALERT_ROUTING_DEFAULT_DEDUPE_WINDOW_SECONDS
            try:
                rate_limit_window_seconds = max(
                    1,
                    min(
                        int(arguments.get("rate_limit_window_seconds", ALERT_ROUTING_DEFAULT_RATE_WINDOW_SECONDS)),
                        7 * 24 * 3600,
                    ),
                )
            except (TypeError, ValueError):
                rate_limit_window_seconds = ALERT_ROUTING_DEFAULT_RATE_WINDOW_SECONDS
            try:
                max_alerts_per_window = max(
                    0,
                    min(
                        int(arguments.get("max_alerts_per_window", ALERT_ROUTING_DEFAULT_MAX_ALERTS_PER_WINDOW)),
                        100000,
                    ),
                )
            except (TypeError, ValueError):
                max_alerts_per_window = ALERT_ROUTING_DEFAULT_MAX_ALERTS_PER_WINDOW
            state_path_value = arguments.get("state_path")
            state_path = Path(str(state_path_value)) if state_path_value else ALERT_ROUTING_STATE_PATH
            try:
                webhook_retry_attempts = max(
                    1,
                    min(
                        int(arguments.get("webhook_retry_attempts", ALERT_ROUTING_DEFAULT_WEBHOOK_RETRY_ATTEMPTS)),
                        10,
                    ),
                )
            except (TypeError, ValueError):
                webhook_retry_attempts = ALERT_ROUTING_DEFAULT_WEBHOOK_RETRY_ATTEMPTS
            try:
                webhook_backoff_ms = max(
                    0,
                    min(
                        int(arguments.get("webhook_backoff_ms", ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MS)),
                        60_000,
                    ),
                )
            except (TypeError, ValueError):
                webhook_backoff_ms = ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MS
            try:
                webhook_backoff_multiplier = max(
                    1.0,
                    min(
                        float(arguments.get("webhook_backoff_multiplier", ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MULTIPLIER)),
                        10.0,
                    ),
                )
            except (TypeError, ValueError):
                webhook_backoff_multiplier = ALERT_ROUTING_DEFAULT_WEBHOOK_BACKOFF_MULTIPLIER
            try:
                webhook_timeout_seconds = max(
                    1,
                    min(
                        int(arguments.get("webhook_timeout_seconds", ALERT_ROUTING_DEFAULT_WEBHOOK_TIMEOUT_SECONDS)),
                        300,
                    ),
                )
            except (TypeError, ValueError):
                webhook_timeout_seconds = ALERT_ROUTING_DEFAULT_WEBHOOK_TIMEOUT_SECONDS
            webhook_retry_status_codes = _sanitize_retry_status_codes(
                arguments.get("webhook_retry_status_codes", ALERT_ROUTING_DEFAULT_RETRY_STATUS_CODES)
            )

            alerts_input = arguments.get("alerts")
            alerts: List[Dict[str, Any]] = []
            context_payload: Dict[str, Any] = {}
            if alerts_input is not None:
                if not isinstance(alerts_input, list):
                    return [types.TextContent(type="text", text="Error: alerts must be an array when provided.")]
                alerts = [a for a in alerts_input if isinstance(a, dict)]
            else:
                job_id = str(arguments.get("job_id") or "").strip()
                if not job_id:
                    return [types.TextContent(
                        type="text",
                        text="Error: Provide `alerts` or `job_id` to derive alerts."
                    )]
                try:
                    lines = max(100, min(int(arguments.get("lines", 500)), 20000))
                except (TypeError, ValueError):
                    lines = 500

                snapshot = await collect_observability_snapshot(
                    client,
                    job_id=job_id,
                    lines=lines,
                    config_name=arguments.get("config_name"),
                    dataset_path_override=arguments.get("dataset_path"),
                    include_dataset=False,
                    include_nlp=False,
                    include_raw_log_excerpt=False,
                    excerpt_lines=80,
                    max_dataset_files=1000,
                )
                if not snapshot.get("success"):
                    return [types.TextContent(
                        type="text",
                        text=f"Failed to derive alerts for job {job_id}: {snapshot.get('error', 'Unknown error')}"
                    )]
                alerts = snapshot.get("alerts", []) or []
                if include_context:
                    context_payload = {
                        "job_id": job_id,
                        "status_snapshot": snapshot.get("status_snapshot"),
                        "overall_health": snapshot.get("overall_health"),
                    }

            filtered_alerts = [
                alert for alert in alerts
                if SEVERITY_ORDER.get(str(alert.get("severity", "none")).lower(), 0) >= threshold
            ]
            state_before = load_alert_routing_state(state_path)
            routable_alerts, filter_stats, next_state = filter_alerts_for_routing(
                filtered_alerts,
                dedupe_window_seconds=dedupe_window_seconds,
                rate_limit_window_seconds=rate_limit_window_seconds,
                max_alerts_per_window=max_alerts_per_window,
                state=state_before,
            )
            delivery_success = False

            route_result = {"routed": 0, "sink": sink, "destination": destination}
            if sink == "file":
                if destination:
                    destination_path = Path(str(destination))
                else:
                    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
                    destination_path = ALERTS_DIR / f"alerts-{_utc_now().strftime('%Y%m%d')}.jsonl"
                destination_path.parent.mkdir(parents=True, exist_ok=True)

                with open(destination_path, "a", encoding="utf-8") as f:
                    for alert in routable_alerts:
                        payload = {
                            "timestamp": _utc_now_iso(),
                            "alert": alert,
                        }
                        if include_context and context_payload:
                            payload["context"] = context_payload
                        f.write(json.dumps(payload) + "\n")

                save_alert_routing_state(state_path, next_state)
                delivery_success = True
                route_result.update({
                    "routed": len(routable_alerts),
                    "destination": str(destination_path),
                })
            elif sink == "webhook":
                if not destination:
                    return [types.TextContent(
                        type="text",
                        text="Error: destination is required for webhook sink."
                    )]
                payload = {"alerts": routable_alerts}
                if include_context and context_payload:
                    payload["context"] = context_payload

                webhook_delivery = await send_webhook_with_retry(
                    str(destination),
                    payload,
                    retry_attempts=webhook_retry_attempts,
                    base_backoff_ms=webhook_backoff_ms,
                    backoff_multiplier=webhook_backoff_multiplier,
                    retry_status_codes=webhook_retry_status_codes,
                    timeout_seconds=webhook_timeout_seconds,
                )
                webhook_status = webhook_delivery.get("status")
                webhook_error = webhook_delivery.get("error")
                webhook_success = bool(webhook_delivery.get("success"))
                webhook_attempts = int(webhook_delivery.get("attempts", 1) or 1)
                webhook_history = webhook_delivery.get("history", [])

                route_result.update({
                    "routed": len(routable_alerts) if webhook_success else 0,
                    "destination": str(destination),
                    "webhook_status": webhook_status,
                    "webhook_error": webhook_error,
                    "webhook_attempts": webhook_attempts,
                    "webhook_retries": max(0, webhook_attempts - 1),
                    "webhook_history": webhook_history,
                })
                if webhook_success:
                    save_alert_routing_state(state_path, next_state)
                    delivery_success = True
            else:
                return [types.TextContent(
                    type="text",
                    text="Error: sink must be either 'file' or 'webhook'."
                )]

            result = {
                "generated_at": _utc_now_iso(),
                "min_severity": min_severity,
                "alerts_in": len(alerts),
                "alerts_after_severity_filter": len(filtered_alerts),
                "alerts_after_dedupe_rate_limit": len(routable_alerts),
                "alerts_routed": route_result.get("routed", 0),
                "routing_controls": {
                    "dedupe_window_seconds": dedupe_window_seconds,
                    "rate_limit_window_seconds": rate_limit_window_seconds,
                    "max_alerts_per_window": max_alerts_per_window,
                    "state_path": str(state_path),
                    "webhook_retry_attempts": webhook_retry_attempts,
                    "webhook_backoff_ms": webhook_backoff_ms,
                    "webhook_backoff_multiplier": webhook_backoff_multiplier,
                    "webhook_timeout_seconds": webhook_timeout_seconds,
                    "webhook_retry_status_codes": webhook_retry_status_codes,
                    "delivery_success": delivery_success,
                },
                "filter_stats": filter_stats,
                "route_result": route_result,
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "list-configs":
            configs = list_configs()
            if configs:
                result = "Available training configurations:\n"
                result += "\n".join([f"- {c}" for c in configs])
            else:
                result = "No training configurations found."
            return [types.TextContent(type="text", text=result)]
            
        elif name == "get-config":
            config_name = arguments.get("name")
            config = load_config(config_name)
            if config:
                result = f"Configuration '{config_name}':\n\n{yaml.dump(config, default_flow_style=False)}"
            else:
                result = f"Configuration '{config_name}' not found."
            return [types.TextContent(type="text", text=result)]
            
        elif name == "get-training-info":
            config_name = arguments.get("name")
            if not config_name:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: name"
                )]
            
            config = load_config(config_name)
            if config and "training_metadata" in config:
                metadata = config["training_metadata"]
                result = f"Training information for '{config_name}':\n"
                result += f"- Trigger word: {metadata.get('trigger_word', 'Not set')}\n"
                result += f"- Model: {metadata.get('model_name', 'Unknown')}\n"
                result += f"- Resolution: {metadata.get('resolution', 'Unknown')}\n"
                result += f"- Steps: {metadata.get('steps', 'Unknown')}\n"
                result += f"- LoRA rank: {metadata.get('rank', 'Unknown')}\n"
                result += f"- Learning rate: {metadata.get('learning_rate', 'Unknown')}\n"
                
                test_prompts = metadata.get('test_prompts', [])
                if test_prompts:
                    result += f"\nTest prompts ({len(test_prompts)}):\n"
                    for i, prompt in enumerate(test_prompts, 1):
                        result += f"{i}. {prompt}\n"
                else:
                    result += "\nNo test prompts configured."
            else:
                result = f"Training information for '{config_name}' not found."
            return [types.TextContent(type="text", text=result)]
            
        elif name == "upload-dataset":
            dataset_name = arguments.get("dataset_name")
            images = arguments.get("images", [])
            
            if not dataset_name:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: dataset_name"
                )]
            
            if not images:
                return [types.TextContent(
                    type="text",
                    text="Error: No images provided. The 'images' parameter must be an array of image objects."
                )]
            
            # Create dataset directory
            dataset_path = DATASET_DIR / dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Save images and captions
            saved_count = 0
            for img_data in images:
                try:
                    # Validate image data structure
                    if not isinstance(img_data, dict):
                        logger.warning(f"Skipping invalid image data: not a dictionary")
                        continue
                    
                    if "filename" not in img_data or "content" not in img_data or "caption" not in img_data:
                        logger.warning(f"Skipping image with missing fields: {img_data.keys()}")
                        continue
                    
                    filename = Path(img_data["filename"]).name
                    image_path = dataset_path / filename
                    caption_path = image_path.with_suffix('.txt')
                    
                    # Save image
                    image_content = base64.b64decode(img_data["content"])
                    with open(image_path, 'wb') as f:
                        f.write(image_content)
                        
                    # Save caption
                    with open(caption_path, 'w') as f:
                        f.write(img_data["caption"])
                        
                    saved_count += 1
                    logger.info(f"Saved image: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error saving image {img_data.get('filename', 'unknown')}: {str(e)}")
                    continue
            
            # Generate .aitk_size.json metadata for AI Toolkit compatibility
            create_aitk_metadata(dataset_path)
            
            result = f"Created dataset '{dataset_name}':\n"
            result += f"- Location: {dataset_path}\n"
            result += f"- Images saved: {saved_count}\n"
            result += f"- Generated .aitk_size.json metadata for AI Toolkit compatibility"
            
            return [types.TextContent(type="text", text=result)]
            
        elif name == "list-datasets":
            datasets = list_datasets()
            if datasets:
                result = "Available datasets:\n"
                for ds in datasets:
                    result += f"- {ds['name']} ({ds['image_count']} images)\n"
            else:
                result = "No datasets found."
            return [types.TextContent(type="text", text=result)]

        elif name == "list-comfyui-models":
            query = str(arguments.get("query", "") or "")
            try:
                limit = int(arguments.get("limit", 50))
            except (TypeError, ValueError):
                limit = 50
            limit = max(1, min(limit, 500))

            roots = get_available_model_roots()
            if not roots:
                return [types.TextContent(
                    type="text",
                    text=(
                        "No local ComfyUI model roots are accessible.\n"
                        f"Configured roots: {', '.join([str(p) for p in COMFYUI_MODEL_ROOTS])}"
                    )
                )]

            models = list_comfyui_models(query=query, limit=limit)
            if not models:
                result = (
                    f"No local ComfyUI models found for query '{query}'.\n"
                    f"Searched roots: {', '.join([str(p) for p in roots])}"
                )
                return [types.TextContent(type="text", text=result)]

            result = (
                f"Local ComfyUI models ({len(models)} shown, limit={limit}):\n"
                f"Searched roots: {', '.join([str(p) for p in roots])}\n"
            )
            for model in models:
                result += f"- {model['name']}\n"
                result += f"  Path: {model['path']}\n"

            return [types.TextContent(type="text", text=result)]
            
        elif name == "start-training":
            config_name = arguments.get("config_name")
            
            if not config_name:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: config_name"
                )]
            
            config_path = CONFIG_DIR / f"{config_name}.yaml"
            
            if not config_path.exists():
                return [types.TextContent(type="text", text=f"Configuration '{config_name}' not found.")]
                
            # Start training via API
            response = await client.start_training(str(config_path))
            
            if response.get("success"):
                job_id = response.get("job_id")
                result = f"Started training job:\n"
                result += f"- Job ID: {job_id}\n"
                result += f"- Configuration: {config_name}\n"
                result += f"- Status: Running"
            else:
                result = f"Failed to start training: {response.get('error', 'Unknown error')}"
                
            return [types.TextContent(type="text", text=result)]
            
        elif name == "get-training-status":
            job_id = arguments.get("job_id")
            
            if not job_id:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: job_id"
                )]
            
            status = await client.get_training_status(job_id)
            
            if status.get("found"):
                result = f"Training job {job_id}:\n"
                result += f"- Status: {status.get('status', 'Unknown')}\n"
                result += f"- Progress: {status.get('progress', 0)}%\n"
                result += f"- Current step: {status.get('current_step', 0)}/{status.get('total_steps', 0)}\n"
                if status.get("eta"):
                    result += f"- ETA: {status.get('eta')}\n"
            else:
                result = f"Training job {job_id} not found."
                
            return [types.TextContent(type="text", text=result)]
            
        elif name == "stop-training":
            job_id = arguments.get("job_id")
            
            if not job_id:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: job_id"
                )]
            
            response = await client.stop_training(job_id)
            
            if response.get("success"):
                result = f"Successfully stopped training job {job_id}"
            else:
                result = f"Failed to stop training: {response.get('error', 'Unknown error')}"
                
            return [types.TextContent(type="text", text=result)]

        elif name == "list-training-jobs":
            response = await client.list_training_jobs()
            if not response.get("success"):
                return [types.TextContent(
                    type="text",
                    text=f"Failed to list training jobs: {response.get('error', 'Unknown error')}"
                )]

            jobs = response.get("jobs", [])
            if not jobs:
                return [types.TextContent(type="text", text="No training jobs found.")]

            result = "Training jobs:\n"
            for job in jobs:
                result += f"- {job.get('name')} ({job.get('id')}): {job.get('status', 'unknown')}"
                if job.get("total_steps", 0):
                    result += f" [{job.get('step', 0)}/{job.get('total_steps', 0)} | {job.get('progress', 0)}%]"
                result += "\n"
            return [types.TextContent(type="text", text=result)]

        elif name == "export-model":
            job_id = arguments.get("job_id")
            export_format = arguments.get("format", "safetensors")

            if not job_id:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: job_id"
                )]

            response = await client.export_model(job_id, export_format)
            if response.get("success"):
                result = "Model export resolved:\n"
                result += f"- Job ID: {job_id}\n"
                result += f"- Format: {response.get('format')}\n"
                result += f"- Model path: {response.get('model_path')}\n"
                result += f"- Size: {response.get('size', 0)} bytes"
            else:
                result = f"Failed to export model: {response.get('error', 'Unknown error')}"

            return [types.TextContent(type="text", text=result)]
            
        elif name == "list-exported-models":
            models = list_output_models()
            if models:
                result = "Available trained models:\n"
                for model in models:
                    size_mb = model['size'] / (1024 * 1024)
                    result += f"- {model['name']}\n"
                    result += f"  Path: {model['path']}\n"
                    result += f"  Size: {size_mb:.2f} MB\n"
                    result += f"  Modified: {model['modified']}\n"
            else:
                result = "No trained models found in outputs directory."
            return [types.TextContent(type="text", text=result)]
            
        elif name == "download-model":
            model_path = arguments.get("model_path")
            include_metadata = arguments.get("include_metadata", True)
            
            if not model_path:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: model_path"
                )]
            
            # Construct full path
            full_path = OUTPUT_DIR / model_path
            
            if not full_path.exists():
                return [types.TextContent(type="text", text=f"Model not found: {model_path}")]
                
            if not full_path.is_file():
                return [types.TextContent(type="text", text=f"Path is not a file: {model_path}")]
                
            # Check file size (warn if > 100MB)
            file_size = full_path.stat().st_size
            if file_size > 100 * 1024 * 1024:
                size_mb = file_size / (1024 * 1024)
                logger.warning(f"Large model file: {size_mb:.2f} MB")
                
            try:
                # Read and encode the model file
                with open(full_path, 'rb') as f:
                    model_content = base64.b64encode(f.read()).decode()
                    
                result = {
                    "filename": full_path.name,
                    "content": model_content,
                    "size": file_size,
                    "path": model_path
                }
                
                # Look for metadata files if requested
                if include_metadata:
                    metadata_files = [
                        full_path.with_suffix('.json'),
                        full_path.with_suffix('.metadata.json'),
                        full_path.parent / f"{full_path.stem}.metadata.json"
                    ]
                    
                    for metadata_path in metadata_files:
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    result["metadata"] = json.load(f)
                                    result["metadata_source"] = metadata_path.name
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to load metadata: {e}")
                                
                    # Check for training config in parent directory
                    config_path = full_path.parent / "config.yaml"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                result["training_config"] = yaml.safe_load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load training config: {e}")
                            
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                logger.error(f"Error downloading model: {e}")
                return [types.TextContent(type="text", text=f"Error downloading model: {str(e)}")]
            
        elif name == "get-system-stats":
            stats = await client.get_system_stats()
            
            result = "System Statistics:\n"
            result += f"- GPU: {stats.get('gpu_name', 'Unknown')}\n"
            result += f"- GPU Memory: {stats.get('gpu_memory_used', 0)}/{stats.get('gpu_memory_total', 0)} MB\n"
            result += f"- GPU Utilization: {stats.get('gpu_utilization', 0)}%\n"
            result += f"- CPU Usage: {stats.get('cpu_percent', 0)}%\n"
            result += f"- RAM Usage: {stats.get('ram_used', 0)}/{stats.get('ram_total', 0)} GB\n"
            result += f"- Active jobs: {stats.get('active_jobs', 0)}"
                
            return [types.TextContent(type="text", text=result)]

        elif name == "get-training-observability":
            job_id = str(arguments.get("job_id") or "").strip()
            if not job_id:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: job_id"
                )]

            lines = arguments.get("lines", 500)
            excerpt_lines = arguments.get("excerpt_lines", 120)
            max_dataset_files = arguments.get("max_dataset_files", 5000)
            include_dataset = bool(arguments.get("include_dataset", True))
            include_nlp = bool(arguments.get("include_nlp", True))
            include_raw_log_excerpt = bool(arguments.get("include_raw_log_excerpt", False))
            include_evaluation = bool(arguments.get("include_evaluation", True))
            include_next_experiment = bool(arguments.get("include_next_experiment", True))
            include_baseline_comparison = bool(arguments.get("include_baseline_comparison", False))

            baseline_job_ids = arguments.get("baseline_job_ids", [])
            if baseline_job_ids is None:
                baseline_job_ids = []
            if not isinstance(baseline_job_ids, list):
                return [types.TextContent(
                    type="text",
                    text="Error: baseline_job_ids must be an array of strings when provided."
                )]

            try:
                lines = max(10, min(int(lines), 20000))
            except (TypeError, ValueError):
                lines = 500
            try:
                excerpt_lines = max(10, min(int(excerpt_lines), 1000))
            except (TypeError, ValueError):
                excerpt_lines = 120
            try:
                max_dataset_files = max(100, min(int(max_dataset_files), 20000))
            except (TypeError, ValueError):
                max_dataset_files = 5000
            try:
                baseline_limit = max(1, min(int(arguments.get("baseline_limit", 5)), 50))
            except (TypeError, ValueError):
                baseline_limit = 5

            pass_threshold = _to_float(arguments.get("pass_threshold"))
            if pass_threshold is None:
                pass_threshold = 70.0
            pass_threshold = float(_clamp(pass_threshold, 0.0, 100.0))

            snapshot = await collect_observability_snapshot(
                client,
                job_id=job_id,
                lines=lines,
                config_name=arguments.get("config_name"),
                dataset_path_override=arguments.get("dataset_path"),
                include_dataset=include_dataset,
                include_nlp=include_nlp,
                include_raw_log_excerpt=include_raw_log_excerpt,
                excerpt_lines=excerpt_lines,
                max_dataset_files=max_dataset_files,
            )
            if not snapshot.get("success"):
                return [types.TextContent(
                    type="text",
                    text=f"Failed to build observability report: {snapshot.get('error', 'Unknown error')}"
                )]

            run_report_response = await collect_job_run_report(
                client,
                job_id,
                log_lines=lines,
                include_dataset=include_dataset,
                include_nlp=include_nlp,
                max_dataset_files=max_dataset_files,
                dataset_cache={},
            )
            run_report = run_report_response.get("run_report", {}) if run_report_response.get("found") else {}
            config_payload = run_report.get("config_payload") if isinstance(run_report, dict) else None

            run_report_public: Dict[str, Any]
            if isinstance(run_report, dict) and run_report:
                run_report_public = dict(run_report)
                run_report_public.pop("config_payload", None)
            else:
                run_report_public = {
                    "job_id": job_id,
                    "error": run_report_response.get("error", "run_report_unavailable"),
                }

            evaluation: Dict[str, Any] = {"skipped": True}
            if include_evaluation:
                if run_report_response.get("found") and isinstance(run_report, dict):
                    evaluation = evaluate_run_report(
                        run_report,
                        pass_threshold=pass_threshold,
                    )
                else:
                    evaluation = {"error": run_report_response.get("error", "run_report_unavailable")}

            next_experiment: Dict[str, Any] = {"skipped": True}
            if include_next_experiment:
                if run_report_response.get("found") and include_evaluation and "overall_score" in evaluation:
                    next_experiment = suggest_next_experiment_from_run(
                        run_report,
                        evaluation,
                        config_payload=config_payload if isinstance(config_payload, dict) else None,
                    )
                else:
                    next_experiment = {
                        "error": "next_experiment_requires_evaluation_and_run_report"
                    }

            intelligence = build_log_intelligence_profile(
                snapshot=snapshot,
                run_report=run_report_public if isinstance(run_report_public, dict) else {},
                evaluation=evaluation if include_evaluation else None,
                config_payload=config_payload if isinstance(config_payload, dict) else None,
            )

            baseline_comparison: Dict[str, Any] = {"skipped": True}
            if include_baseline_comparison:
                candidate_ids = [
                    str(job).strip()
                    for job in baseline_job_ids
                    if str(job).strip() and str(job).strip() != job_id
                ]
                candidate_ids = list(dict.fromkeys(candidate_ids))

                if not candidate_ids:
                    jobs_resp = await client.list_training_jobs()
                    if jobs_resp.get("success"):
                        for job in jobs_resp.get("jobs", []):
                            candidate = str(job.get("id") or "").strip()
                            if not candidate or candidate == job_id:
                                continue
                            candidate_ids.append(candidate)
                            if len(candidate_ids) >= baseline_limit:
                                break

                dataset_cache: Dict[str, Tuple[Dict[str, Any], List[str]]] = {}
                comparison_reports: List[Dict[str, Any]] = []
                skipped_baselines: List[Dict[str, Any]] = []
                target_run_found = bool(run_report_response.get("found") and isinstance(run_report, dict))

                if target_run_found and isinstance(run_report_public, dict):
                    comparison_reports.append(run_report_public)

                for baseline_job_id in candidate_ids[:baseline_limit]:
                    baseline_resp = await collect_job_run_report(
                        client,
                        baseline_job_id,
                        log_lines=lines,
                        include_dataset=include_dataset,
                        include_nlp=include_nlp,
                        max_dataset_files=max_dataset_files,
                        dataset_cache=dataset_cache,
                    )
                    if not baseline_resp.get("found"):
                        skipped_baselines.append({
                            "job_id": baseline_job_id,
                            "reason": baseline_resp.get("error", "collection_failed"),
                        })
                        continue
                    baseline_report = baseline_resp.get("run_report", {})
                    if isinstance(baseline_report, dict):
                        baseline_public = dict(baseline_report)
                        baseline_public.pop("config_payload", None)
                        comparison_reports.append(baseline_public)

                if not target_run_found:
                    baseline_comparison = {
                        "enabled": True,
                        "error": "target_run_unavailable",
                        "target_job_id": job_id,
                        "candidate_job_ids": candidate_ids[:baseline_limit],
                        "collected_baseline_job_ids": [report.get("job_id") for report in comparison_reports],
                        "skipped_baselines": skipped_baselines,
                    }
                elif len(comparison_reports) >= 2:
                    ranking_payload = rank_run_comparisons(comparison_reports)
                    baseline_comparison = {
                        "enabled": True,
                        "compared_job_ids": [report.get("job_id") for report in comparison_reports],
                        "skipped_baselines": skipped_baselines,
                        "position": summarize_ranking_position(ranking_payload, job_id),
                        "ranking": ranking_payload,
                    }
                else:
                    baseline_comparison = {
                        "enabled": True,
                        "error": "insufficient_baseline_runs",
                        "candidate_job_ids": candidate_ids[:baseline_limit],
                        "skipped_baselines": skipped_baselines,
                    }

            report = dict(snapshot)
            report.pop("success", None)
            report["analysis_mode"] = "deep_log_intelligence"
            report["inputs"] = {
                "lines": lines,
                "include_dataset": include_dataset,
                "include_nlp": include_nlp,
                "include_raw_log_excerpt": include_raw_log_excerpt,
                "excerpt_lines": excerpt_lines,
                "max_dataset_files": max_dataset_files,
                "include_evaluation": include_evaluation,
                "include_next_experiment": include_next_experiment,
                "include_baseline_comparison": include_baseline_comparison,
                "baseline_limit": baseline_limit,
                "baseline_job_ids": baseline_job_ids,
                "pass_threshold": pass_threshold,
            }
            report["run_report"] = run_report_public
            report["evaluation"] = evaluation if include_evaluation else {"skipped": True}
            report["intelligence"] = intelligence
            report["next_experiment"] = next_experiment if include_next_experiment else {"skipped": True}
            report["baseline_comparison"] = baseline_comparison

            return [types.TextContent(type="text", text=json.dumps(report, indent=2))]

        elif name == "get-training-logs":
            job_id = arguments.get("job_id")
            lines = arguments.get("lines", 100)

            if not job_id:
                return [types.TextContent(
                    type="text",
                    text="Error: Missing required parameter: job_id"
                )]

            response = await client.get_training_logs(job_id, lines)
            if response.get("success"):
                log_text = response.get("log", "")
                if not log_text:
                    result = f"No logs available for job {job_id}."
                else:
                    result = f"Training logs for {job_id}:\n{log_text}"
            else:
                result = f"Failed to get training logs: {response.get('error', 'Unknown error')}"

            return [types.TextContent(type="text", text=result)]
            
        # Add more tool implementations as needed...
            
    except Exception as e:
        logger.error(f"Error handling tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    finally:
        await client.disconnect_websocket()

async def main():
    """Main entry point for the MCP server"""
    logger.info("Configured ComfyUI model roots: %s", [str(p) for p in COMFYUI_MODEL_ROOTS])
    logger.info("Accessible ComfyUI model roots: %s", [str(p) for p in get_available_model_roots()])
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="ai-toolkit-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
