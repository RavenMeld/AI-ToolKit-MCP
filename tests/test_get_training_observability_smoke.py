import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

from mcp_server import DELTA_TO_WINNER_BAND_SOURCE_ID, handle_call_tool  # noqa: E402


class GetTrainingObservabilitySmokeTests(unittest.TestCase):
    def test_smoke_get_training_observability_with_mock_backend(self):
        class DummyClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def get_training_status(self, job_id):
                if job_id == "job-baseline":
                    return {
                        "found": True,
                        "status": "completed",
                        "current_step": 1000,
                        "total_steps": 1000,
                        "progress": 100.0,
                        "speed_string": "0.25 it/s",
                    }
                return {
                    "found": True,
                    "status": "running",
                    "current_step": 240,
                    "total_steps": 1000,
                    "progress": 24.0,
                    "speed_string": "0.03 it/s",
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

            async def get_training_logs(self, job_id, _lines=500):
                if job_id == "job-baseline":
                    log_text = "\n".join(
                        [
                            "step=10 loss: 0.82 speed 0.22 it/s",
                            "step=20 loss: 0.71 speed 0.24 it/s",
                            "step=30 loss: 0.63 speed 0.25 it/s",
                        ]
                    )
                else:
                    log_text = "\n".join(
                        [
                            "step=10 loss: 1.20 speed 0.03 it/s",
                            "step=20 loss: 1.10 speed 0.03 it/s",
                            "step=30 loss: 1.04 speed 0.04 it/s",
                        ]
                    )
                return {"success": True, "log": log_text}

            async def list_training_jobs(self):
                return {"success": True, "jobs": []}

            async def get_system_stats(self):
                return {
                    "gpu_utilization": 68,
                    "gpu_memory_used": 9340,
                    "gpu_memory_total": 24564,
                    "cpu_percent": 32,
                    "ram_used": 14.2,
                    "ram_total": 62.0,
                }

            async def disconnect_websocket(self):
                return None

        with patch("mcp_server.AIToolkitClient", DummyClient):
            result = asyncio.run(
                handle_call_tool(
                    "get-training-observability",
                    {
                        "job_id": "job-smoke",
                        "lines": 300,
                        "include_dataset": False,
                        "include_nlp": False,
                        "include_evaluation": True,
                        "include_next_experiment": True,
                        "include_baseline_comparison": True,
                        "baseline_job_ids": ["job-baseline"],
                        "baseline_limit": 2,
                    },
                )
            )

        self.assertEqual(len(result), 1)
        payload = json.loads(result[0].text)

        self.assertEqual(payload["analysis_mode"], "deep_log_intelligence")
        self.assertIn("intelligence", payload)
        self.assertIn("baseline_comparison", payload)
        self.assertIn("alerts", payload)
        self.assertIsInstance(payload["alerts"], list)

        intelligence = payload["intelligence"]
        self.assertIn("scores", intelligence)
        self.assertIn("decision_gates", intelligence)
        self.assertIn("thresholds", intelligence)
        self.assertIn("confidence", intelligence)
        self.assertTrue(len(intelligence.get("bottlenecks", [])) > 0)
        self.assertIn("confidence_source", intelligence["bottlenecks"][0])

        baseline = payload["baseline_comparison"]
        self.assertTrue(baseline["enabled"])
        self.assertIn("ranking", baseline)
        ranking_items = baseline["ranking"]["ranking"]
        self.assertEqual(len(ranking_items), 2)
        self.assertIn("delta_to_winner", ranking_items[0])
        self.assertIn("band_source", ranking_items[0]["delta_to_winner"])
        self.assertEqual(
            ranking_items[0]["delta_to_winner"]["band_source"]["id"],
            DELTA_TO_WINNER_BAND_SOURCE_ID,
        )


if __name__ == "__main__":
    unittest.main()
