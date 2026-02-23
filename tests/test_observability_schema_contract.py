import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import types

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


class ObservabilitySchemaContractTests(unittest.TestCase):
    def test_get_training_observability_output_schema_sections(self):
        class DummyClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def list_training_jobs(self):
                return {"success": True, "jobs": []}

            async def disconnect_websocket(self):
                return None

        async def fake_collect_observability_snapshot(*_args, **_kwargs):
            return {
                "success": True,
                "job_id": "job-target",
                "overall_health": "healthy",
                "speed_analysis": {"status": "healthy", "iter_per_sec": {"mean": 0.18, "last": 0.18}},
                "quality_analysis": {"status": "healthy", "trend": "improving", "flags": [], "convergence_score": 82.0},
                "error_analysis": {"severity": "none", "total_mentions": 0},
                "dataset_analysis": {"exists": True, "caption_coverage_pct": 97, "duplicate_image_estimate": 0},
                "nlp_analysis": {"caption_token_diversity": 0.41, "style_flags": []},
                "alerts": [],
            }

        run_values = {
            "job-target": (82.0, 0.18, 85.0, 90.0),
            "job-a": (90.0, 0.26, 92.0, 92.0),
            "job-b": (70.0, 0.10, 74.0, 80.0),
        }

        async def fake_collect_job_run_report(_client, job_id, **_kwargs):
            conv, speed, stability, dataset = run_values[job_id]
            return {
                "found": True,
                "run_report": {
                    "job_id": job_id,
                    "metrics": {
                        "convergence_score": conv,
                        "speed_mean_iter_per_sec": speed,
                        "stability_score": stability,
                        "dataset_quality_score": dataset,
                    },
                    "quality_analysis": {"status": "healthy", "flags": []},
                    "speed_analysis": {"status": "healthy", "iter_per_sec": {"mean": speed}},
                    "error_analysis": {"severity": "none", "total_mentions": 0},
                    "dataset_analysis": {"exists": True, "caption_coverage_pct": 97},
                    "nlp_analysis": {"caption_token_diversity": 0.41, "style_flags": []},
                    "alerts": [],
                    "config_payload": {
                        "config": {
                            "process": [
                                {
                                    "model": {"name_or_path": "runwayml/stable-diffusion-v1-5", "arch": "sd15"},
                                    "train": {"steps": 3200, "batch_size": 1, "lr": 0.0002},
                                    "network": {"linear": 16, "linear_alpha": 16},
                                    "datasets": [{"resolution": [768]}],
                                }
                            ]
                        }
                    },
                },
            }

        with patch("mcp_server.AIToolkitClient", DummyClient), patch(
            "mcp_server.collect_observability_snapshot",
            side_effect=fake_collect_observability_snapshot,
        ), patch(
            "mcp_server.collect_job_run_report",
            side_effect=fake_collect_job_run_report,
        ):
            result = asyncio.run(
                handle_call_tool(
                    "get-training-observability",
                    {
                        "job_id": "job-target",
                        "lines": 250,
                        "include_dataset": True,
                        "include_nlp": True,
                        "include_evaluation": True,
                        "include_next_experiment": True,
                        "include_baseline_comparison": True,
                        "baseline_job_ids": ["job-a", "job-b"],
                        "baseline_limit": 2,
                    },
                )
            )

        self.assertEqual(len(result), 1)
        payload = json.loads(result[0].text)

        top_keys = {
            "analysis_mode",
            "inputs",
            "run_report",
            "evaluation",
            "intelligence",
            "next_experiment",
            "baseline_comparison",
            "speed_analysis",
            "quality_analysis",
            "error_analysis",
            "dataset_analysis",
            "nlp_analysis",
            "alerts",
        }
        self.assertTrue(top_keys.issubset(payload.keys()))
        self.assertEqual(payload["analysis_mode"], "deep_log_intelligence")
        self.assertIsInstance(payload["alerts"], list)

        inputs = payload["inputs"]
        for key in (
            "lines",
            "include_dataset",
            "include_nlp",
            "include_evaluation",
            "include_next_experiment",
            "include_baseline_comparison",
            "baseline_limit",
            "baseline_job_ids",
        ):
            self.assertIn(key, inputs)

        run_report = payload["run_report"]
        for key in ("job_id", "metrics", "quality_analysis", "speed_analysis", "error_analysis", "dataset_analysis", "nlp_analysis", "alerts"):
            self.assertIn(key, run_report)

        evaluation = payload["evaluation"]
        for key in ("overall_score", "grade", "passed", "pass_threshold", "components", "weights", "diagnostics"):
            self.assertIn(key, evaluation)

        intelligence = payload["intelligence"]
        for key in ("scores", "bottlenecks", "recommendations", "decision_gates", "thresholds", "context", "confidence"):
            self.assertIn(key, intelligence)
        speed_curve = intelligence["context"].get("speed_curve_thresholds", {})
        for key in ("excellent", "good", "fair", "watch", "poor"):
            self.assertIn(key, speed_curve)

        next_experiment = payload["next_experiment"]
        for key in ("recommendation_type", "parameter", "current", "proposed", "expected_impact", "risk_level", "rationale", "validation_plan"):
            self.assertIn(key, next_experiment)

        baseline = payload["baseline_comparison"]
        for key in ("enabled", "compared_job_ids", "skipped_baselines", "position", "ranking"):
            self.assertIn(key, baseline)
        self.assertEqual(baseline["compared_job_ids"], ["job-target", "job-a", "job-b"])

        ranking = baseline["ranking"]
        for key in ("ranking", "winner_job_id", "winner_score", "score_spread"):
            self.assertIn(key, ranking)
        self.assertEqual(ranking["winner_job_id"], "job-a")
        self.assertEqual(len(ranking["ranking"]), 3)

        for item in ranking["ranking"]:
            self.assertIn("job_id", item)
            self.assertIn("scores", item)
            self.assertIn("rank", item)
            self.assertIn("delta_to_winner", item)
            delta = item["delta_to_winner"]
            for key in ("total", "total_band", "components", "component_bands", "band_source"):
                self.assertIn(key, delta)
            band_source = delta["band_source"]
            for key in ("id", "units", "thresholds"):
                self.assertIn(key, band_source)
            self.assertEqual(band_source["id"], DELTA_TO_WINNER_BAND_SOURCE_ID)

    def test_get_training_observability_skips_optional_sections_when_disabled(self):
        class DummyClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def list_training_jobs(self):
                return {"success": True, "jobs": []}

            async def disconnect_websocket(self):
                return None

        async def fake_collect_observability_snapshot(*_args, **_kwargs):
            return {
                "success": True,
                "job_id": "job-target",
                "overall_health": "healthy",
                "speed_analysis": {"status": "healthy", "iter_per_sec": {"mean": 0.2, "last": 0.2}},
                "quality_analysis": {"status": "healthy", "trend": "improving", "flags": [], "convergence_score": 84.0},
                "error_analysis": {"severity": "none", "total_mentions": 0},
                "dataset_analysis": {"exists": True, "caption_coverage_pct": 98, "duplicate_image_estimate": 0},
                "nlp_analysis": {"caption_token_diversity": 0.45, "style_flags": []},
                "alerts": [],
            }

        async def fake_collect_job_run_report(_client, _job_id, **_kwargs):
            return {
                "found": True,
                "run_report": {
                    "job_id": "job-target",
                    "metrics": {
                        "convergence_score": 84.0,
                        "speed_mean_iter_per_sec": 0.2,
                        "stability_score": 90.0,
                        "dataset_quality_score": 92.0,
                    },
                    "quality_analysis": {"status": "healthy", "flags": []},
                    "speed_analysis": {"status": "healthy", "iter_per_sec": {"mean": 0.2}},
                    "error_analysis": {"severity": "none", "total_mentions": 0},
                    "dataset_analysis": {"exists": True, "caption_coverage_pct": 98},
                    "nlp_analysis": {"caption_token_diversity": 0.45, "style_flags": []},
                    "alerts": [],
                },
            }

        with patch("mcp_server.AIToolkitClient", DummyClient), patch(
            "mcp_server.collect_observability_snapshot",
            side_effect=fake_collect_observability_snapshot,
        ), patch(
            "mcp_server.collect_job_run_report",
            side_effect=fake_collect_job_run_report,
        ):
            result = asyncio.run(
                handle_call_tool(
                    "get-training-observability",
                    {
                        "job_id": "job-target",
                        "lines": 200,
                        "include_dataset": True,
                        "include_nlp": True,
                        "include_evaluation": False,
                        "include_next_experiment": False,
                        "include_baseline_comparison": False,
                    },
                )
            )

        self.assertEqual(len(result), 1)
        payload = json.loads(result[0].text)
        self.assertEqual(payload.get("analysis_mode"), "deep_log_intelligence")

        inputs = payload.get("inputs", {})
        self.assertFalse(inputs.get("include_evaluation"))
        self.assertFalse(inputs.get("include_next_experiment"))
        self.assertFalse(inputs.get("include_baseline_comparison"))

        self.assertEqual(payload.get("evaluation"), {"skipped": True})
        self.assertEqual(payload.get("next_experiment"), {"skipped": True})
        self.assertEqual(payload.get("baseline_comparison"), {"skipped": True})

        intelligence = payload.get("intelligence", {})
        for key in ("scores", "bottlenecks", "recommendations", "decision_gates", "thresholds", "context", "confidence"):
            self.assertIn(key, intelligence)

    def test_get_training_observability_bottlenecks_include_confidence_source(self):
        class DummyClient:
            def __init__(self, *_args, **_kwargs):
                pass

            async def list_training_jobs(self):
                return {"success": True, "jobs": []}

            async def disconnect_websocket(self):
                return None

        async def fake_collect_observability_snapshot(*_args, **_kwargs):
            return {
                "success": True,
                "job_id": "job-target",
                "overall_health": "warning",
                "speed_analysis": {"status": "slow", "iter_per_sec": {"mean": 0.03, "last": 0.03}},
                "quality_analysis": {"status": "warning", "trend": "regressing", "flags": [], "convergence_score": 66.0},
                "error_analysis": {"severity": "none", "total_mentions": 0},
                "dataset_analysis": {"exists": True, "caption_coverage_pct": 98, "duplicate_image_estimate": 0},
                "nlp_analysis": {"caption_token_diversity": 0.45, "style_flags": []},
                "alerts": [
                    {"severity": "warning", "category": "quality", "message": "regressing", "confidence": 0.78},
                    {"severity": "warning", "category": "speed", "message": "slow", "confidence": 0.76},
                ],
            }

        async def fake_collect_job_run_report(_client, _job_id, **_kwargs):
            return {
                "found": True,
                "run_report": {
                    "job_id": "job-target",
                    "metrics": {
                        "convergence_score": 66.0,
                        "speed_mean_iter_per_sec": 0.03,
                        "stability_score": 84.0,
                        "dataset_quality_score": 92.0,
                    },
                    "quality_analysis": {"status": "warning", "trend": "regressing", "flags": []},
                    "speed_analysis": {"status": "slow", "iter_per_sec": {"mean": 0.03}},
                    "error_analysis": {"severity": "none", "total_mentions": 0},
                    "dataset_analysis": {"exists": True, "caption_coverage_pct": 98},
                    "nlp_analysis": {"caption_token_diversity": 0.45, "style_flags": []},
                    "alerts": [
                        {"severity": "warning", "category": "quality", "message": "regressing", "confidence": 0.78},
                        {"severity": "warning", "category": "speed", "message": "slow", "confidence": 0.76},
                    ],
                },
            }

        with patch("mcp_server.AIToolkitClient", DummyClient), patch(
            "mcp_server.collect_observability_snapshot",
            side_effect=fake_collect_observability_snapshot,
        ), patch(
            "mcp_server.collect_job_run_report",
            side_effect=fake_collect_job_run_report,
        ):
            result = asyncio.run(
                handle_call_tool(
                    "get-training-observability",
                    {
                        "job_id": "job-target",
                        "lines": 200,
                        "include_dataset": True,
                        "include_nlp": True,
                        "include_evaluation": False,
                        "include_next_experiment": False,
                        "include_baseline_comparison": False,
                    },
                )
            )

        self.assertEqual(len(result), 1)
        payload = json.loads(result[0].text)
        intelligence = payload.get("intelligence", {})
        bottlenecks = intelligence.get("bottlenecks", [])
        self.assertGreaterEqual(len(bottlenecks), 1)
        allowed_sources = {"alert_category_max", "severity_default"}
        has_alert_derived = False

        for item in bottlenecks:
            self.assertIn("confidence", item)
            self.assertIn("confidence_source", item)
            self.assertIn(item["confidence_source"], allowed_sources)
            if item["confidence_source"] == "alert_category_max":
                has_alert_derived = True

        self.assertTrue(has_alert_derived)


if __name__ == "__main__":
    unittest.main()
