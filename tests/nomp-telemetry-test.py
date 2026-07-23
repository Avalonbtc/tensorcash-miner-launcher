"""Focused tests for the source-owned NOMP vLLM telemetry helper.

The launcher copies the overlay into TensorCash's runtime source, so loading
the full module would require the complete proxy dependency graph.  This test
executes only the parser and pure fixed-window telemetry definitions extracted
from the overlay itself.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path
from typing import Any
import unittest


OVERLAY = Path(__file__).resolve().parents[1] / "nomp-sidecar-overlay.py"


def load_telemetry_definitions():
    tree = ast.parse(OVERLAY.read_text(encoding="utf-8"), filename=str(OVERLAY))
    wanted = {
        "VLLM_METRICS_WINDOW_SECONDS",
        "VLLM_METRICS_POLL_SECONDS",
        "VLLM_METRICS_TIMEOUT_SECONDS",
        "_parse_vllm_generation_metrics",
        "_VllmGenerationTelemetry",
    }
    body = []
    for node in tree.body:
        names = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        if any(name in wanted for name in names):
            body.append(node)
    namespace = {"Any": Any, "collections": collections}
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, str(OVERLAY), "exec"), namespace)
    return namespace


class VllmTelemetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = load_telemetry_definitions()

    def test_parser_does_not_double_count_tensor_parallel_labels(self):
        parse = self.definitions["_parse_vllm_generation_metrics"]
        payload = """
        vllm:generation_tokens_total{engine=\"0\"} 1200
        vllm:generation_tokens_total{engine=\"1\"} 1200
        vllm:num_requests_running{engine=\"0\"} 32
        vllm:num_requests_running{engine=\"1\"} 32
        vllm:num_requests_waiting{engine=\"0\"} 7
        """
        self.assertEqual(parse(payload), (1200.0, 32, 7))

    def test_counter_rate_is_continuous_without_completed_requests(self):
        telemetry = self.definitions["_VllmGenerationTelemetry"](window_seconds=60.0)
        telemetry.record(timestamp=0.0, generation_tokens=100.0, running=960, waiting=0)
        telemetry.record(timestamp=10.0, generation_tokens=3100.0, running=960, waiting=0)
        snapshot = telemetry.snapshot()
        self.assertTrue(snapshot["ready"])
        self.assertEqual(snapshot["generation_tokens_per_sec"], 300.0)
        self.assertEqual(snapshot["running"], 960)

    def test_counter_reset_never_becomes_a_negative_rate(self):
        telemetry = self.definitions["_VllmGenerationTelemetry"](window_seconds=60.0)
        telemetry.record(timestamp=0.0, generation_tokens=1000.0, running=64, waiting=0)
        telemetry.record(timestamp=10.0, generation_tokens=2000.0, running=64, waiting=0)
        telemetry.record(timestamp=20.0, generation_tokens=50.0, running=0, waiting=0)
        self.assertFalse(telemetry.snapshot()["ready"])
        telemetry.record(timestamp=30.0, generation_tokens=1050.0, running=64, waiting=0)
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["generation_tokens_per_sec"], 100.0)
        self.assertEqual(snapshot["counter_resets"], 1)


if __name__ == "__main__":
    unittest.main()
