"""Authenticated local work-unit API for a NOMP TensorCash client.

It converts a NOMP work unit into the official ``MineRequest`` representation,
injects it into the existing VDF/proof pipeline, and returns only proof data
emitted by ``ProofCollector``.  It never synthesizes a hash, nonce, or proof.
"""

from __future__ import annotations

import base64
import asyncio
import collections
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from components.mining_protocol import (
    MINING_MODE_DUMMY_ONLY,
    MINING_MODE_REQUEST_ATTACHED,
    SUBMIT_POLICY_RETURN_TO_CLIENT,
    MineRequest,
    MiningModelMeta,
    MiningPolicy,
    MiningProtocolError,
    MiningTemplate,
)
from components.proof_collector import (
    _extract_is_solution,
    _extract_model_identifier,
    _extract_proof_hash_hex,
    _extract_proof_nonce,
    _extract_req_id,
)


logger = logging.getLogger(__name__)


def _bounded_positive_env(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer tuning knob without accepting unsafe values."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


@dataclass
class _JobState:
    request: MineRequest
    expires_at: int
    created_at: float = field(default_factory=time.time)
    # Never use deque(maxlen=...) here: silently discarding a valid proof is a
    # revenue loss.  The scheduler applies a high-water mark before it starts
    # more inference; the bounded overshoot is at most the active request set.
    results: collections.deque[dict[str, Any]] = field(default_factory=collections.deque)
    cancelled: bool = False
    block_pending: bool = False
    backpressured: bool = False


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MiningProtocolError(f"NOMP {field} must be a non-empty string")
    return value.strip()


def _hex(value: Any, field: str, length: int) -> str:
    raw = _required_string(value, field).removeprefix("0x").removeprefix("0X")
    if len(raw) != length:
        raise MiningProtocolError(f"NOMP {field} must be {length} hexadecimal characters")
    try:
        bytes.fromhex(raw)
    except ValueError as exc:
        raise MiningProtocolError(f"NOMP {field} is not hexadecimal") from exc
    return raw.lower()


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MiningProtocolError(f"NOMP {field} must be an integer") from exc
    if parsed < 1:
        raise MiningProtocolError(f"NOMP {field} must be positive")
    return parsed


def build_nomp_mine_request(payload: Any, *, max_parallel: int = 1) -> tuple[MineRequest, int]:
    """Validate the NOMP job and convert it to the sidecar's typed form."""
    if not isinstance(payload, dict):
        raise MiningProtocolError("NOMP work unit must be a JSON object")
    if payload.get("protocol") != "tensorcash-stratum/1":
        raise MiningProtocolError(f"unsupported NOMP protocol: {payload.get('protocol')!r}")

    job_id = _required_string(payload.get("job_id"), "job_id")
    request_id = _positive_int(payload.get("request_id", payload.get("work_unit_id")), "request_id")
    work_unit_id = _positive_int(payload.get("work_unit_id", request_id), "work_unit_id")
    if request_id != work_unit_id:
        raise MiningProtocolError("NOMP request_id and work_unit_id must match")
    expires_at = _positive_int(payload.get("expires_at"), "expires_at")
    if expires_at <= int(time.time()):
        raise MiningProtocolError("NOMP work unit is already expired")

    header_prefix = _hex(payload.get("header_prefix"), "header_prefix", 152)
    target = _hex(payload.get("target"), "target", 64)
    base_share_target = _hex(payload.get("base_share_target"), "base_share_target", 64)
    tip_hash = _hex(payload.get("tip_hash"), "tip_hash", 64)
    model_raw = payload.get("model")
    if not isinstance(model_raw, dict):
        raise MiningProtocolError("NOMP model must be an object")
    model = MiningModelMeta(
        name=_required_string(model_raw.get("name"), "model.name"),
        commit=_required_string(model_raw.get("commit"), "model.commit"),
        model_hash=(
            _hex(model_raw["model_hash"], "model.model_hash", 64)
            if model_raw.get("model_hash")
            else None
        ),
        difficulty=_positive_int(model_raw.get("difficulty"), "model.difficulty"),
    )
    mode = str(payload.get("mode") or MINING_MODE_DUMMY_ONLY)
    if mode not in (MINING_MODE_DUMMY_ONLY, MINING_MODE_REQUEST_ATTACHED):
        raise MiningProtocolError(f"unsupported NOMP mining mode: {mode!r}")
    bits = int.from_bytes(bytes.fromhex(header_prefix)[72:76], "little")
    request = MineRequest(
        job_id=job_id,
        work_unit_id=work_unit_id,
        wallet_id="nomp-local-sidecar",
        network=_required_string(payload.get("network"), "network"),
        mode=mode,
        model=model,
        template=MiningTemplate(
            template_id=job_id,
            request_id=request_id,
            block_hash=tip_hash,
            header_prefix=header_prefix,
            target=target,
            bits=bits,
            expires_at=expires_at,
            base_share_target=base_share_target,
        ),
        # This is local scheduling metadata only.  The registered model,
        # header, target, VDF, and proof payload stay exactly the same.
        policy=MiningPolicy(
            submit_policy=SUBMIT_POLICY_RETURN_TO_CLIENT,
            max_parallel=max_parallel,
        ),
    )
    return request, expires_at


def _build_block_header_fb(request: MineRequest) -> bytes:
    """Use the same header conversion as the official broker worker."""
    import flatbuffers
    from proof import BlockHeader

    prefix = bytes.fromhex(request.template.header_prefix)
    builder = flatbuffers.Builder(1024)
    prev_hash = builder.CreateByteVector(prefix[4:36])
    merkle_root = builder.CreateByteVector(prefix[36:68])
    BlockHeader.BlockHeaderStart(builder)
    BlockHeader.BlockHeaderAddVersion(builder, int.from_bytes(prefix[0:4], "little"))
    BlockHeader.BlockHeaderAddPrevBlockHash(builder, prev_hash)
    BlockHeader.BlockHeaderAddMerkleRoot(builder, merkle_root)
    BlockHeader.BlockHeaderAddTimestamp(builder, int.from_bytes(prefix[68:72], "little"))
    BlockHeader.BlockHeaderAddBits(builder, int.from_bytes(prefix[72:76], "little"))
    BlockHeader.BlockHeaderAddReqId(builder, request.template.request_id)
    header = BlockHeader.BlockHeaderEnd(builder)
    builder.Finish(header)
    return bytes(builder.Output())


class NompSidecarController:
    """Owns one proof-producing local NOMP work unit at a time."""

    def __init__(self, context, zmq_listener, proof_collector, request_manager):
        self.context = context
        self.zmq_listener = zmq_listener
        self.proof_collector = proof_collector
        self.request_manager = request_manager
        self.token = os.getenv("NOMP_SIDECAR_TOKEN", "").strip()
        if len(self.token) < 16:
            raise RuntimeError(
                "NOMP_SIDECAR_TOKEN must be set to a secret of at least 16 characters"
            )
        self.parallelism = _bounded_positive_env(
            "NOMP_SIDECAR_CONCURRENCY", default=1, minimum=1, maximum=8
        )
        self.min_buffered_proofs = _bounded_positive_env(
            "NOMP_SIDECAR_MIN_BUFFERED_PROOFS", default=2, minimum=1, maximum=64
        )
        self.max_buffered_proofs = _bounded_positive_env(
            "NOMP_SIDECAR_MAX_BUFFERED_PROOFS", default=8, minimum=1, maximum=64
        )
        if self.min_buffered_proofs >= self.max_buffered_proofs:
            raise RuntimeError(
                "NOMP_SIDECAR_MIN_BUFFERED_PROOFS must be smaller than "
                "NOMP_SIDECAR_MAX_BUFFERED_PROOFS"
            )
        if self.max_buffered_proofs < self.parallelism:
            raise RuntimeError(
                "NOMP_SIDECAR_MAX_BUFFERED_PROOFS must be at least "
                "NOMP_SIDECAR_CONCURRENCY"
            )
        vllm_max_seqs = _bounded_positive_env(
            "VLLM_MAX_NUM_SEQS", default=1, minimum=1, maximum=1024
        )
        if self.parallelism > vllm_max_seqs:
            raise RuntimeError(
                "NOMP_SIDECAR_CONCURRENCY cannot exceed VLLM_MAX_NUM_SEQS; "
                f"got {self.parallelism} > {vllm_max_seqs}"
            )
        self._lock = threading.RLock()
        self._jobs_by_id: dict[str, _JobState] = {}
        self._job_id_by_request: dict[int, str] = {}
        self._mine_tasks: dict[str, set[asyncio.Task]] = {}
        # ProofCollector runs on its own thread.  Capture the aiohttp loop on
        # the first submit so proof callbacks can schedule a refill safely.
        self._loop: asyncio.AbstractEventLoop | None = None
        self.proof_collector.set_solution_callback(self._on_proof)

    def _authorized(self, request: web.Request) -> bool:
        return request.headers.get("Authorization", "") == f"Bearer {self.token}"

    def _active_model_matches(self, mine: MineRequest) -> bool:
        active = self.request_manager.get_active_model()
        return (
            active.get("model_name") == mine.model.name
            and active.get("model_commit") == mine.model.commit
            and not active.get("switch_in_progress")
        )

    async def close(self) -> None:
        """Stop local dummy generation before the proxy HTTP session closes."""
        with self._lock:
            tasks = self._retire_active_jobs_locked()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _retire_active_jobs_locked(self) -> list[asyncio.Task]:
        """Invalidate every active lease and return its tasks for cancellation.

        This is intentionally separate from awaiting task cancellation: an old
        vLLM completion may take a short time to observe cancellation, but it
        must stop being eligible to publish a proof before a replacement job is
        admitted.  `_on_proof` checks both the mapping and `cancelled` flag.
        """
        tasks = [task for job_tasks in self._mine_tasks.values() for task in job_tasks]
        self._mine_tasks.clear()
        for state in self._jobs_by_id.values():
            state.cancelled = True
        self._jobs_by_id.clear()
        self._job_id_by_request.clear()
        return tasks

    def _prune_expired(self) -> None:
        now = int(time.time())
        for job_id, state in list(self._jobs_by_id.items()):
            if state.expires_at <= now or state.cancelled:
                self._jobs_by_id.pop(job_id, None)
                self._job_id_by_request.pop(state.request.template.request_id, None)
                for task in self._mine_tasks.pop(job_id, set()):
                    task.cancel()

    def _active_task_count_locked(self, job_id: str) -> int:
        tasks = self._mine_tasks.get(job_id, set())
        finished = {task for task in tasks if task.done()}
        tasks.difference_update(finished)
        if not tasks:
            self._mine_tasks.pop(job_id, None)
            return 0
        return len(tasks)

    def _cancel_mining_tasks_locked(self, job_id: str) -> None:
        for task in self._mine_tasks.get(job_id, set()):
            task.cancel()

    def _ensure_mining_locked(self, job_id: str) -> None:
        """Keep a bounded number of genuine inference requests in flight.

        The high/low watermark prevents a slow pool or verifier from consuming
        unbounded RAM, while normal share-rate verification still leaves all
        configured vLLM slots busy.  It never replaces a proof with a newer
        one, and it stops new inference immediately after a block candidate.
        """
        state = self._jobs_by_id.get(job_id)
        if (
            not state
            or state.cancelled
            or state.block_pending
            or state.expires_at <= int(time.time())
        ):
            return

        buffered = len(state.results)
        if state.backpressured:
            if buffered > self.min_buffered_proofs:
                return
            state.backpressured = False
            logger.info(
                "NOMP work %s resumed below proof buffer low-water mark (%d)",
                job_id,
                self.min_buffered_proofs,
            )
        if buffered >= self.max_buffered_proofs:
            state.backpressured = True
            logger.warning(
                "NOMP work %s reached proof buffer high-water mark (%d); pausing new inference",
                job_id,
                self.max_buffered_proofs,
            )
            return

        tasks = self._mine_tasks.setdefault(job_id, set())
        self._active_task_count_locked(job_id)
        tasks = self._mine_tasks.setdefault(job_id, set())
        while len(tasks) < self.parallelism:
            task = asyncio.create_task(self._mine_once(job_id))
            tasks.add(task)
            task.add_done_callback(
                lambda finished, work_id=job_id: self._on_mine_task_done(work_id, finished)
            )

    async def _mine_once(self, job_id: str) -> None:
        """Run exactly one genuine vLLM request for a live NOMP lease."""
        with self._lock:
            state = self._jobs_by_id.get(job_id)
            if (
                not state
                or state.cancelled
                or state.block_pending
                or state.backpressured
                or state.expires_at <= int(time.time())
            ):
                return
            model_name = state.request.model.name
        try:
            await self.request_manager.generate_nomp_dummy(model_name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Avoid a tight error loop if vLLM is restarting.  The done
            # callback refills the slot after this bounded retry delay.
            logger.warning("NOMP dummy request failed for %s: %s", job_id, exc)
            await asyncio.sleep(1)

    def _on_mine_task_done(self, job_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return
        with self._lock:
            self._active_task_count_locked(job_id)
            self._ensure_mining_locked(job_id)

    def _schedule_refill_from_callback(self, job_id: str) -> None:
        with self._lock:
            self._ensure_mining_locked(job_id)

    async def submit(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="missing or invalid sidecar token")
        try:
            mine, expires_at = build_nomp_mine_request(
                await request.json(), max_parallel=self.parallelism
            )
        except (MiningProtocolError, ValueError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        if not self._active_model_matches(mine):
            active = self.request_manager.get_active_model()
            raise web.HTTPConflict(
                text=(
                    "sidecar model does not match NOMP work unit; active="
                    f"{active.get('model_name')}@{active.get('model_commit')}"
                )
            )
        tasks_to_cancel: list[asyncio.Task] = []
        replaced_count = 0
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._prune_expired()
            if mine.job_id in self._jobs_by_id:
                return web.json_response(
                    {"ok": True, "job_id": mine.job_id, "status": "mining", "reused": True}
                )
            if self._jobs_by_id:
                # A miner reconnect or a genuine new template may arrive while
                # a previous local lease is still winding down.  Returning 429
                # here made the Rust worker discard the new work and then idle
                # until a later NOMP notify.  Retire the old lease atomically;
                # any late proof is ignored by `_on_proof`, and the new lease
                # becomes the sole proof-producing work unit immediately.
                replaced_count = len(self._jobs_by_id)
                tasks_to_cancel = self._retire_active_jobs_locked()
            self.request_manager.set_nomp_model_config(
                name=mine.model.name,
                commit=mine.model.commit,
                model_hash=mine.model.model_hash or "",
                difficulty=mine.model.difficulty or 0,
            )
            self.context.set_expected_model_identifier(mine.model.name, mine.model.commit)
            self.zmq_listener._process_mining_job(
                _build_block_header_fb(mine), base_share_target=mine.template.base_share_target
            )
            self._jobs_by_id[mine.job_id] = _JobState(mine, expires_at)
            self._job_id_by_request[mine.template.request_id] = mine.job_id
            self._ensure_mining_locked(mine.job_id)
        for task in tasks_to_cancel:
            task.cancel()
        if replaced_count:
            logger.info(
                "NOMP sidecar replaced %d superseded local work unit(s) for new request %s",
                replaced_count,
                mine.template.request_id,
            )
        return web.json_response(
            {
                "ok": True,
                "job_id": mine.job_id,
                "status": "mining",
                "parallelism": self.parallelism,
                "buffered_proofs": 0,
            }
        )

    async def status(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="missing or invalid sidecar token")
        job_id = request.match_info["job_id"]
        with self._lock:
            self._prune_expired()
            state = self._jobs_by_id.get(job_id)
            if not state:
                # A controller may be polling while a lease expires or a newer
                # NOMP notify cancels it.  Keep the 404 semantics, but make the
                # terminal state machine-readable rather than returning an HTML
                # or plain-text aiohttp error page.
                return web.json_response(
                    {
                        "ok": False,
                        "job_id": job_id,
                        "status": "expired",
                        "error": "unknown, expired, or cancelled work unit",
                    },
                    status=404,
                )
            task_count = self._active_task_count_locked(job_id)
            common = {
                "ok": True,
                "job_id": job_id,
                "parallelism": self.parallelism,
                "inflight": task_count,
                "buffered_proofs": len(state.results),
                "buffer_limit": self.max_buffered_proofs,
            }
            if not state.results:
                return web.json_response({**common, "status": "mining"})
            return web.json_response({**common, "status": "proof", **state.results[0]})

    async def acknowledge(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="missing or invalid sidecar token")
        with self._lock:
            state = self._jobs_by_id.get(request.match_info["job_id"])
            if not state:
                # Acknowledge is intentionally idempotent.  If the sidecar has
                # already pruned a lease, the pool's terminal reply has still
                # been observed by the controller and no retry is useful.
                return web.json_response(
                    {"ok": True, "acknowledged": False, "status": "expired"}
                )
            acknowledged = state.results.popleft() if state.results else None
            # A block candidate changes the chain template as soon as it is
            # accepted.  Keep any already-buffered proofs visible for the
            # controller to settle, but never start another old-template run.
            if acknowledged and acknowledged.get("is_block"):
                state.block_pending = True
                self._cancel_mining_tasks_locked(request.match_info["job_id"])
            if not state.results and state.block_pending:
                state.cancelled = True
                self._job_id_by_request.pop(state.request.template.request_id, None)
                self._jobs_by_id.pop(request.match_info["job_id"], None)
                for task in self._mine_tasks.pop(request.match_info["job_id"], set()):
                    task.cancel()
            else:
                self._ensure_mining_locked(request.match_info["job_id"])
        return web.json_response({"ok": True})

    async def cancel(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="missing or invalid sidecar token")
        with self._lock:
            state = self._jobs_by_id.pop(request.match_info["job_id"], None)
            if state:
                state.cancelled = True
                self._job_id_by_request.pop(state.request.template.request_id, None)
            for task in self._mine_tasks.pop(request.match_info["job_id"], set()):
                task.cancel()
        return web.json_response({"ok": True, "cancelled": bool(state)})

    def _on_proof(self, req_id: int, proof: bytes) -> None:
        """Called on ProofCollector's thread after official proof construction."""
        with self._lock:
            self._prune_expired()
            job_id = self._job_id_by_request.get(req_id)
            state = self._jobs_by_id.get(job_id or "")
            if (
                not state
                or state.cancelled
                or state.block_pending
                or _extract_req_id(proof) != req_id
            ):
                return
            achieved_hash = _extract_proof_hash_hex(proof)
            nonce = _extract_proof_nonce(proof)
            if not achieved_hash or nonce is None:
                return
            result = {
                "proof_b64": base64.b64encode(proof).decode("ascii"),
                "nonce": str(nonce),
                "achieved_hash": achieved_hash,
                "model_identifier": _extract_model_identifier(proof) or "",
                "is_block": bool(_extract_is_solution(proof)),
            }
            if result["is_block"]:
                # Block propagation beats ordinary share accounting.  A block
                # is never silently dropped behind a buffered share.
                state.block_pending = True
                state.results.appendleft(result)
                self._cancel_mining_tasks_locked(job_id)
            else:
                state.results.append(result)
                if len(state.results) > self.max_buffered_proofs + self.parallelism:
                    logger.error(
                        "NOMP proof buffer overshot its bound for %s: buffered=%d max=%d inflight=%d",
                        job_id,
                        len(state.results),
                        self.max_buffered_proofs,
                        self.parallelism,
                    )
            loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._schedule_refill_from_callback, job_id)
