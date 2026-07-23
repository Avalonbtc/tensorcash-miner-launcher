#!/usr/bin/env python3
"""Regression tests for a cancelled or stuck vLLM recovery probe."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_overlay() -> types.ModuleType:
    """Load the overlay with only the dependency surface these tests require."""
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(
        Request=object,
        Response=object,
        HTTPUnauthorized=RuntimeError,
        HTTPBadRequest=RuntimeError,
        HTTPNotFound=RuntimeError,
        HTTPConflict=RuntimeError,
        HTTPTooManyRequests=RuntimeError,
        json_response=lambda *args, **kwargs: None,
    )
    sys.modules["aiohttp"] = aiohttp

    components = types.ModuleType("components")
    components.__path__ = []
    sys.modules["components"] = components

    protocol = types.ModuleType("components.mining_protocol")
    protocol.MINING_MODE_DUMMY_ONLY = "dummy_only"
    protocol.MINING_MODE_REQUEST_ATTACHED = "request_attached"
    protocol.SUBMIT_POLICY_RETURN_TO_CLIENT = "return_to_client"
    protocol.MiningProtocolError = RuntimeError
    protocol.MineRequest = object
    protocol.MiningModelMeta = object
    protocol.MiningPolicy = object
    protocol.MiningTemplate = object
    sys.modules["components.mining_protocol"] = protocol

    collector = types.ModuleType("components.proof_collector")
    collector._extract_is_solution = lambda *_: False
    collector._extract_model_identifier = lambda *_: ""
    collector._extract_proof_hash_hex = lambda *_: ""
    collector._extract_proof_nonce = lambda *_: 0
    collector._extract_req_id = lambda *_: 0
    sys.modules["components.proof_collector"] = collector

    spec = importlib.util.spec_from_file_location(
        "nomp_sidecar_overlay_test", ROOT / "nomp-sidecar-overlay.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _controller(module: types.ModuleType, request_manager: object):
    controller = module.NompSidecarController.__new__(module.NompSidecarController)
    state = types.SimpleNamespace(
        cancelled=False,
        block_pending=False,
        backpressured=False,
        expires_at=int(time.time()) + 60,
        waiting_for_vdf=False,
        vllm_paused=False,
        request=types.SimpleNamespace(model=types.SimpleNamespace(name="Qwen/Qwen3-8B")),
    )
    controller._lock = threading.RLock()
    controller._jobs_by_id = {"job": state}
    controller._mine_tasks = {}
    controller._vllm_circuit_open = False
    controller._vllm_circuit_until = 0.0
    controller._vllm_failure_streak = 0
    controller._vllm_probe_inflight = True
    controller._vllm_initial_backoff_seconds = 1
    controller._vllm_max_backoff_seconds = 30
    controller._vllm_recovery_probe_timeout_seconds = 0.01
    controller._mining_errors_total = 0
    controller._last_mining_error = ""
    controller._last_mining_error_unix_ms = 0
    controller.request_manager = request_manager
    controller.context = types.SimpleNamespace(
        read=lambda: types.SimpleNamespace(vdf_proof="checkpoint")
    )
    return controller


class _HangingRequestManager:
    async def generate_nomp_dummy(self, _model_name: str) -> None:
        await asyncio.sleep(3600)


async def _check_timeout(module: types.ModuleType) -> None:
    controller = _controller(module, _HangingRequestManager())
    await controller._mine_once("job", 0.0, recovery_probe=True)
    assert controller._vllm_circuit_open
    assert not controller._vllm_probe_inflight
    assert controller._mining_errors_total == 1


async def _check_cancellation(module: types.ModuleType) -> None:
    controller = _controller(module, _HangingRequestManager())
    task = asyncio.create_task(controller._mine_once("job", 0.0, recovery_probe=True))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert not controller._vllm_probe_inflight


def main() -> None:
    module = _load_overlay()
    module.logger.disabled = True
    asyncio.run(_check_timeout(module))
    asyncio.run(_check_cancellation(module))
    print("sidecar recovery tests: OK")


if __name__ == "__main__":
    main()
