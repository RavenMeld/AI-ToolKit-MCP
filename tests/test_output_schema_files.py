import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "schemas"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test environment fallback stubs when optional runtime deps are absent locally.
try:
    import aiohttp  # noqa: F401
except ModuleNotFoundError:
    aiohttp_stub = types.ModuleType("aiohttp")

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    aiohttp_stub.ClientSession = _DummyClientSession
    sys.modules["aiohttp"] = aiohttp_stub

try:
    import websockets  # noqa: F401
except ModuleNotFoundError:
    websockets_stub = types.ModuleType("websockets")

    async def _dummy_connect(*_args, **_kwargs):
        raise RuntimeError("websockets unavailable in test environment")

    websockets_stub.connect = _dummy_connect
    sys.modules["websockets"] = websockets_stub

from mcp_server import handle_call_tool  # noqa: E402


def _load_schema(name: str) -> dict:
    schema_path = SCHEMA_DIR / name
    with open(schema_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _schema_errors(schema_name: str, payload: dict) -> list[str]:
    schema = _load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    formatted = []
    for err in errors:
        path = ".".join(str(item) for item in err.absolute_path) or "<root>"
        formatted.append(f"{path}: {err.message}")
    return formatted


class OutputSchemaFileValidationTests(unittest.TestCase):
    def test_get_training_observability_matches_schema_file(self):
        class DummyClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def get_training_status(self, _job_id):
                return {
                    "found": True,
                    "status": "running",
                    "current_step": 120,
                    "total_steps": 1000,
                    "progress": 12.0,
                    "speed_string": "0.12 it/s",
                }

            async def get_training_job(self, job_id):
                config_payload = {
                    "config": {
                        "process": [
                            {
                                "model": {"name_or_path": "runwayml/stable-diffusion-v1-5", "arch": "sd15"},
                                "train": {"steps": 1000, "batch_size": 1, "lr": 0.0002},
                                "network": {"linear": 16, "linear_alpha": 16},
                                "datasets": [{"resolution": [768]}],
                            }
                        ]
                    }
                }
                return {
                    "found": True,
                    "job": {
                        "id": job_id,
                        "name": f"run-{job_id}",
                        "status": "running",
                        "job_config": json.dumps(config_payload),
                    },
                }

            async def get_training_logs(self, _job_id, _lines=500):
                return {
                    "success": True,
                    "log": "\n".join(
                        [
                            "step=10 loss: 1.20 speed 0.11 it/s",
                            "step=20 loss: 1.05 speed 0.12 it/s",
                            "step=30 loss: 0.97 speed 0.13 it/s",
                        ]
                    ),
                }

            async def get_system_stats(self):
                return {
                    "gpu_utilization": 62,
                    "gpu_memory_used": 9100,
                    "gpu_memory_total": 24564,
                    "cpu_percent": 31,
                    "ram_used": 12.4,
                    "ram_total": 62.0,
                }

            async def list_training_jobs(self):
                return {"success": True, "jobs": []}

            async def disconnect_websocket(self):
                return None

        with patch("mcp_server.AIToolkitClient", DummyClient):
            result = asyncio.run(
                handle_call_tool(
                    "get-training-observability",
                    {
                        "job_id": "job-schema-1",
                        "lines": 300,
                        "include_dataset": False,
                        "include_nlp": False,
                        "include_evaluation": True,
                        "include_next_experiment": True,
                        "include_baseline_comparison": False,
                    },
                )
            )

        self.assertEqual(len(result), 1)
        payload = json.loads(result[0].text)
        errors = _schema_errors("get-training-observability.schema.json", payload)
        self.assertEqual(errors, [], msg="Schema validation failed:\n" + "\n".join(errors))

    def test_compare_training_runs_matches_schema_file(self):
        class DummyClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def get_training_status(self, _job_id):
                return {
                    "found": True,
                    "status": "completed",
                    "current_step": 1000,
                    "total_steps": 1000,
                    "progress": 100.0,
                    "speed_string": "0.2 it/s",
                }

            async def disconnect_websocket(self):
                return None

        async def fake_collect_job_run_report(_client, job_id, **_kwargs):
            metrics_by_job = {
                "job-a": (88.0, 0.22, 90.0, 92.0),
                "job-b": (72.0, 0.12, 74.0, 80.0),
            }
            convergence, speed, stability, dataset = metrics_by_job[job_id]
            return {
                "found": True,
                "run_report": {
                    "job_id": job_id,
                    "metrics": {
                        "convergence_score": convergence,
                        "speed_mean_iter_per_sec": speed,
                        "stability_score": stability,
                        "dataset_quality_score": dataset,
                    },
                    "quality_analysis": {"status": "healthy", "flags": []},
                    "speed_analysis": {"status": "healthy", "iter_per_sec": {"mean": speed}},
                    "error_analysis": {"severity": "none", "total_mentions": 0},
                    "dataset_analysis": {"exists": True, "caption_coverage_pct": 98},
                    "nlp_analysis": {"caption_token_diversity": 0.45, "style_flags": []},
                    "alerts": [],
                },
            }

        with patch("mcp_server.AIToolkitClient", DummyClient), patch(
            "mcp_server.collect_job_run_report",
            side_effect=fake_collect_job_run_report,
        ):
            result = asyncio.run(
                handle_call_tool(
                    "compare-training-runs",
                    {
                        "job_ids": ["job-a", "job-b"],
                        "include_dataset": True,
                        "include_nlp": True,
                    },
                )
            )

        self.assertEqual(len(result), 1)
        payload = json.loads(result[0].text)
        errors = _schema_errors("compare-training-runs.schema.json", payload)
        self.assertEqual(errors, [], msg="Schema validation failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
