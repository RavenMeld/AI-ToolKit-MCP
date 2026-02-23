import sys
import unittest
from pathlib import Path
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

from mcp_server import (  # noqa: E402
    DELTA_TO_WINNER_BAND_SOURCE_ID,
    DELTA_TO_WINNER_MODERATE_MAX,
    DELTA_TO_WINNER_NEGLIGIBLE_MAX,
    rank_run_comparisons,
    score_dataset_quality,
    score_stability,
    summarize_timeseries_points,
)


class TimeseriesAndComparisonHelperTests(unittest.TestCase):
    def test_summarize_timeseries_points(self):
        points = [
            {
                "current_step": 10,
                "iter_per_sec": 0.2,
                "loss_last": 1.2,
                "gpu_utilization_pct": 60,
                "gpu_memory_used_mb": 8000,
                "status": "running",
                "overall_health": "healthy",
                "alert_count": 0,
                "alert_max_severity": "none",
            },
            {
                "current_step": 40,
                "iter_per_sec": 0.25,
                "loss_last": 0.9,
                "gpu_utilization_pct": 75,
                "gpu_memory_used_mb": 9200,
                "status": "running",
                "overall_health": "warning",
                "alert_count": 2,
                "alert_max_severity": "warning",
            },
        ]
        summary = summarize_timeseries_points(points)
        self.assertEqual(summary["point_count"], 2)
        self.assertEqual(summary["step"]["delta"], 30)
        self.assertGreater(summary["iter_per_sec"]["mean"], 0.0)
        self.assertEqual(summary["alerts"]["total_alerts"], 2)

    def test_score_dataset_quality(self):
        good = {
            "exists": True,
            "scanned_images": 100,
            "caption_coverage_pct": 98,
            "duplicate_image_estimate": 0,
            "corrupt_image_count": 0,
            "low_resolution_count": 2,
            "extreme_aspect_ratio_count": 1,
        }
        bad = {
            "exists": True,
            "scanned_images": 100,
            "caption_coverage_pct": 50,
            "duplicate_image_estimate": 25,
            "corrupt_image_count": 3,
            "low_resolution_count": 40,
            "extreme_aspect_ratio_count": 20,
        }
        good_score = score_dataset_quality(good)
        bad_score = score_dataset_quality(bad)
        self.assertIsNotNone(good_score)
        self.assertIsNotNone(bad_score)
        self.assertGreater(good_score, bad_score)

    def test_score_stability(self):
        stable = score_stability(
            quality_analysis={"status": "healthy", "flags": []},
            error_analysis={"severity": "none", "total_mentions": 0},
            alerts=[],
        )
        unstable = score_stability(
            quality_analysis={"status": "critical", "flags": ["loss_nan_detected"]},
            error_analysis={"severity": "critical", "total_mentions": 10},
            alerts=[{"severity": "critical"}, {"severity": "warning"}],
        )
        self.assertGreater(stable, unstable)

    def test_rank_run_comparisons(self):
        reports = [
            {
                "job_id": "job_a",
                "metrics": {
                    "convergence_score": 70,
                    "speed_mean_iter_per_sec": 0.1,
                    "stability_score": 65,
                    "dataset_quality_score": 80,
                },
            },
            {
                "job_id": "job_b",
                "metrics": {
                    "convergence_score": 85,
                    "speed_mean_iter_per_sec": 0.25,
                    "stability_score": 90,
                    "dataset_quality_score": 88,
                },
            },
            {
                "job_id": "job_c",
                "metrics": {
                    "convergence_score": 40,
                    "speed_mean_iter_per_sec": 0.05,
                    "stability_score": 30,
                    "dataset_quality_score": 50,
                },
            },
        ]
        ranking = rank_run_comparisons(reports)
        self.assertEqual(ranking["winner_job_id"], "job_b")
        self.assertEqual(ranking["ranking"][0]["job_id"], "job_b")
        self.assertGreater(ranking["score_spread"], 0.0)
        self.assertIn("delta_to_winner", ranking["ranking"][0])
        self.assertEqual(ranking["ranking"][0]["delta_to_winner"]["total"], 0.0)
        self.assertEqual(ranking["ranking"][0]["delta_to_winner"]["total_band"], "negligible")
        self.assertIn("band_source", ranking["ranking"][0]["delta_to_winner"])
        self.assertEqual(
            ranking["ranking"][0]["delta_to_winner"]["band_source"]["id"],
            DELTA_TO_WINNER_BAND_SOURCE_ID,
        )
        thresholds = ranking["ranking"][0]["delta_to_winner"]["band_source"]["thresholds"]
        self.assertEqual(thresholds["negligible_lte"], DELTA_TO_WINNER_NEGLIGIBLE_MAX)
        self.assertEqual(thresholds["moderate_lte"], DELTA_TO_WINNER_MODERATE_MAX)
        self.assertLess(ranking["ranking"][1]["delta_to_winner"]["total"], 0.0)
        self.assertIn("component_bands", ranking["ranking"][1]["delta_to_winner"])


if __name__ == "__main__":
    unittest.main()
