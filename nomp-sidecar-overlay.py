"""Authenticated local work-unit API for a NOMP TensorCash client.

It converts a NOMP work unit into the official ``MineRequest`` representation,
injects it into the existing VDF/proof pipeline, and returns only proof data
emitted by ``ProofCollector``.  It never synthesizes a hash, nonce, or proof.
"""

from __future__ import annotations

import base64
import asyncio
import collections
import hashlib
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

# The native launcher permits 96/128 only for a 24 GiB TP=1 profile with a
# deliberately larger vLLM KV-cache reservation. Keep the sidecar's ceiling
# aligned with that profile: a typo such as 320 would otherwise create hundreds
# of concurrent HTTP coroutines, exceed the available KV cache, inflate
# stale-work cancellation, and harm sustained generation throughput.
MAX_NOMP_SIDECAR_CONCURRENCY = 128
VDF_NOT_READY_MESSAGE = "VDF proof not yet available"


def _bounded_positive_env(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer tuning knob without accepting unsafe values."""
    raw = os.getenv(name, str(default)).strip()
    return _bounded_positive(raw, name, minimum, maximum)


def _bounded_positive(raw: str, name: str, minimum: int, maximum: int) -> int:
    """Validate one positive integer value, including a non-environment input."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _bounded_nonnegative_env(name: str, default: int, maximum: int) -> int:
    """Read an optional non-negative millisecond tuning knob safely."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if not 0 <= value <= maximum:
        raise RuntimeError(f"{name} must be between 0 and {maximum}, got {value}")
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
    # Parallel inference can occasionally reach ProofCollector twice with the
    # exact same proof bytes.  A fresh local proof_id is not enough to make
    # that a new share: NOMP correctly identifies the duplicate by its proof
    # payload.  Keep compact digests for the life of this work unit so only
    # genuinely distinct proofs ever reach the controller/pool.
    proof_fingerprints: set[bytes] = field(default_factory=set)
    duplicate_proofs_dropped: int = 0
    # A controller claims a proof before it sends it to the pool. Claims are
    # short-lived so a controller crash cannot strand revenue, while normal
    # operation can submit several proofs concurrently without resending the
    # queue head over and over.
    claims: dict[int, float] = field(default_factory=dict)
    next_proof_id: int = 1
    # A deterministic sequence used to slightly stagger otherwise identical
    # 256-token requests. Without it, a large batch starts and ends together,
    # producing a periodic GPU-utilization valley even though work is queued.
    next_admission_slot: int = 0
    cancelled: bool = False
    block_pending: bool = False
    backpressured: bool = False
    # A fresh block has no usable VDF proof until the local prover reaches its
    # first checkpoint. This is a normal transition, not a failed inference.
    waiting_for_vdf: bool = False


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
        vllm_max_seqs = _bounded_positive_env(
            "VLLM_MAX_NUM_SEQS", default=1, minimum=1, maximum=1024
        )
        parallelism_raw = os.getenv("NOMP_SIDECAR_CONCURRENCY", "auto").strip().lower()
        self.adaptive_enabled = parallelism_raw in {"", "auto"}
        if self.adaptive_enabled:
            adaptive_ceiling = min(vllm_max_seqs, MAX_NOMP_SIDECAR_CONCURRENCY)
            requested_minimum = _bounded_positive_env(
                "NOMP_SIDECAR_ADAPTIVE_MIN_CONCURRENCY",
                default=4,
                minimum=1,
                maximum=MAX_NOMP_SIDECAR_CONCURRENCY,
            )
            self.adaptive_min_parallelism = min(requested_minimum, adaptive_ceiling)
            requested_maximum = _bounded_positive_env(
                "NOMP_SIDECAR_ADAPTIVE_MAX_CONCURRENCY",
                default=adaptive_ceiling,
                minimum=1,
                maximum=MAX_NOMP_SIDECAR_CONCURRENCY,
            )
            self.max_parallelism = max(
                self.adaptive_min_parallelism,
                min(requested_maximum, adaptive_ceiling),
            )
            requested_start = _bounded_positive_env(
                "NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY",
                default=min(32, self.max_parallelism),
                minimum=1,
                maximum=MAX_NOMP_SIDECAR_CONCURRENCY,
            )
            self.parallelism = min(
                self.max_parallelism,
                max(self.adaptive_min_parallelism, requested_start),
            )
            self.adaptive_step = _bounded_positive_env(
                "NOMP_SIDECAR_ADAPTIVE_STEP",
                default=16,
                minimum=1,
                maximum=64,
            )
            self.adaptive_interval_seconds = _bounded_positive_env(
                "NOMP_SIDECAR_ADAPTIVE_INTERVAL_SECONDS",
                default=60,
                minimum=30,
                maximum=300,
            )
        else:
            self.parallelism = _bounded_positive(
                parallelism_raw,
                "NOMP_SIDECAR_CONCURRENCY",
                minimum=1,
                maximum=MAX_NOMP_SIDECAR_CONCURRENCY,
            )
            self.adaptive_min_parallelism = self.parallelism
            self.max_parallelism = self.parallelism
            self.adaptive_step = 0
            self.adaptive_interval_seconds = 0
        if self.parallelism > vllm_max_seqs:
            raise RuntimeError(
                "NOMP sidecar concurrency cannot exceed VLLM_MAX_NUM_SEQS; "
                f"got {self.parallelism} > {vllm_max_seqs}"
            )
        # Keep a small reserve request queue inside vLLM.  Fixed-length PoW
        # completions otherwise tend to return in a cohort, briefly leaving
        # the engine with no runnable work while aiohttp callbacks refill the
        # sidecar.  These are queued requests, not extra GPU sequences.
        # Keep this opt-in. Some vLLM builds schedule fixed-length decode
        # cohorts more efficiently without queued requests; a reserve must
        # never become a default throughput regression.
        default_prefetch_requests = 0
        self.prefetch_requests = _bounded_nonnegative_env(
            "NOMP_SIDECAR_PREFETCH_REQUESTS",
            default=default_prefetch_requests,
            maximum=64,
        )
        default_min_buffered = max(2, min(64, self.parallelism // 2))
        default_max_buffered = min(
            256, max(8, (self.max_parallelism + self.prefetch_requests) * 2)
        )
        self.min_buffered_proofs = _bounded_positive_env(
            "NOMP_SIDECAR_MIN_BUFFERED_PROOFS",
            default=default_min_buffered,
            minimum=1,
            maximum=128,
        )
        self.max_buffered_proofs = _bounded_positive_env(
            "NOMP_SIDECAR_MAX_BUFFERED_PROOFS",
            default=default_max_buffered,
            minimum=1,
            maximum=256,
        )
        self.claim_lease_seconds = _bounded_positive_env(
            "TENSORCASH_NOMP_CLAIM_LEASE_SECONDS", default=60, minimum=10, maximum=300
        )
        if self.min_buffered_proofs >= self.max_buffered_proofs:
            raise RuntimeError(
                "NOMP_SIDECAR_MIN_BUFFERED_PROOFS must be smaller than "
                "NOMP_SIDECAR_MAX_BUFFERED_PROOFS"
            )
        if self.max_buffered_proofs < self.max_scheduler_parallelism:
            raise RuntimeError(
                "NOMP_SIDECAR_MAX_BUFFERED_PROOFS must be at least "
                "the maximum adaptive concurrency plus NOMP_SIDECAR_PREFETCH_REQUESTS"
            )
        # Keep the first and subsequent vLLM admissions slightly out of phase.
        # At 96/128 slots this is less than one inference duration, so it does
        # not reduce steady-state occupancy; it prevents all same-length jobs
        # from completing and refilling in a single burst.
        default_admission_spread_ms = (
            0
            if self.parallelism <= 4
            else min(1_000, max(160, self.parallelism * 6))
        )
        self._admission_spread_is_automatic = not os.getenv(
            "NOMP_SIDECAR_ADMISSION_SPREAD_MS", ""
        ).strip()
        self.admission_spread_ms = _bounded_nonnegative_env(
            "NOMP_SIDECAR_ADMISSION_SPREAD_MS",
            default=default_admission_spread_ms,
            maximum=2_000,
        )
        self._adaptive_last_evaluation_at = 0.0
        self._adaptive_last_error_count = 0
        self._adaptive_probe_from: int | None = None
        self._adaptive_probe_rate: float | None = None
        self._adaptive_last_action = "starting"
        self._mining_errors_total = 0
        self._last_mining_error = ""
        self._last_mining_error_unix_ms = 0
        logger.info(
            "NOMP scheduler configured: parallelism=%d adaptive=%s range=%d-%d step=%d prefetch=%d scheduler_target=%d admission_spread_ms=%d",
            self.parallelism,
            self.adaptive_enabled,
            self.adaptive_min_parallelism,
            self.max_parallelism,
            self.adaptive_step,
            self.prefetch_requests,
            self.scheduler_parallelism,
            self.admission_spread_ms,
        )
        self._lock = threading.RLock()
        self._jobs_by_id: dict[str, _JobState] = {}
        self._job_id_by_request: dict[int, str] = {}
        self._mine_tasks: dict[str, set[asyncio.Task]] = {}
        # ProofCollector runs on its own thread.  Capture the aiohttp loop on
        # the first submit so proof callbacks can schedule a refill safely.
        self._loop: asyncio.AbstractEventLoop | None = None
        self.proof_collector.set_solution_callback(self._on_proof)

    @property
    def scheduler_parallelism(self) -> int:
        return self.parallelism + self.prefetch_requests

    @property
    def max_scheduler_parallelism(self) -> int:
        return self.max_parallelism + self.prefetch_requests

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

    async def cancel_active(self) -> None:
        """Compatibility lifecycle hook used by the pinned proxy runtime.

        The public runtime image registers this coroutine during application
        shutdown.  Keep the old hook name as a strict alias for ``close`` so a
        scheduler-overlay upgrade cannot turn an otherwise healthy vLLM
        process into a restart loop before it accepts any NOMP work.
        """
        await self.close()

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

    def _next_admission_delay_locked(self, state: _JobState) -> float:
        """Return a bounded deterministic launch offset for one mine task."""
        if self.parallelism <= 1 or self.admission_spread_ms <= 0:
            return 0.0
        slot = state.next_admission_slot % self.parallelism
        state.next_admission_slot += 1
        return (slot * self.admission_spread_ms) / (self.parallelism - 1) / 1_000.0

    def _set_parallelism_locked(self, target: int, action: str) -> bool:
        """Change only future admissions; completed work is never cancelled."""
        target = max(self.adaptive_min_parallelism, min(target, self.max_parallelism))
        if target == self.parallelism:
            self._adaptive_last_action = action
            return False
        previous = self.parallelism
        self.parallelism = target
        if self._admission_spread_is_automatic:
            self.admission_spread_ms = (
                0 if target <= 4 else min(1_000, max(160, target * 6))
            )
        self._adaptive_last_action = action
        logger.info(
            "NOMP adaptive concurrency changed %d -> %d (%s)",
            previous,
            target,
            action,
        )
        return True

    def _maybe_adjust_parallelism_locked(self, status: dict[str, Any] | None = None) -> None:
        """Run one conservative throughput probe at most once per sample window.

        The sidecar starts at 32 requests, then probes one higher bounded level.
        A candidate is retained only when its rolling completion rate is at least
        2% better than the previous level; a 5% regression or any request error
        immediately returns to the known-good level. This intentionally tunes
        useful generation throughput, not a one-second GPU-utilisation spike.
        """
        if not self.adaptive_enabled:
            return
        if any(state.waiting_for_vdf for state in self._jobs_by_id.values()):
            self._adaptive_last_action = "waiting for first VDF checkpoint after block change"
            return
        now = time.monotonic()
        if now - self._adaptive_last_evaluation_at < self.adaptive_interval_seconds:
            return
        self._adaptive_last_evaluation_at = now
        if status is None:
            status = self.request_manager.get_status()
        throughput = status.get("throughput", {}) if isinstance(status, dict) else {}
        rate = float(throughput.get("completion_tokens_per_sec", 0.0) or 0.0)
        window_seconds = float(throughput.get("window_seconds", 0.0) or 0.0)
        active_requests = int(status.get("active_requests", 0) or 0)
        error_delta = self._mining_errors_total - self._adaptive_last_error_count
        self._adaptive_last_error_count = self._mining_errors_total

        if error_delta:
            fallback = self._adaptive_probe_from
            if fallback is None:
                fallback = max(self.adaptive_min_parallelism, self.parallelism - self.adaptive_step)
            self._set_parallelism_locked(fallback, f"rollback after {error_delta} request error(s)")
            self._adaptive_probe_from = None
            self._adaptive_probe_rate = None
            return
        if window_seconds < self.adaptive_interval_seconds * 0.75 or rate <= 0.0:
            self._adaptive_last_action = "waiting for a full throughput window"
            return

        if self._adaptive_probe_from is not None and self._adaptive_probe_rate is not None:
            baseline = self._adaptive_probe_rate
            previous = self._adaptive_probe_from
            self._adaptive_probe_from = None
            self._adaptive_probe_rate = None
            required_active = max(1, int(self.parallelism * 0.75))
            if active_requests < required_active:
                self._set_parallelism_locked(
                    previous,
                    f"rollback: vLLM admitted {active_requests}/{self.parallelism} requests",
                )
                return
            if rate < baseline * 0.95:
                self._set_parallelism_locked(
                    previous,
                    f"rollback: {rate:.1f} tok/s below {baseline:.1f} tok/s baseline",
                )
                return
            if rate >= baseline * 1.02:
                self._adaptive_last_action = (
                    f"kept probe: {rate:.1f} tok/s vs {baseline:.1f} tok/s"
                )
                return
            self._set_parallelism_locked(
                previous,
                f"rollback: probe gain below 2% ({rate:.1f} vs {baseline:.1f} tok/s)",
            )
            return

        if active_requests < max(1, int(self.parallelism * 0.75)):
            self._adaptive_last_action = "holding: vLLM has not filled the current target"
            return

        if self.parallelism >= self.max_parallelism:
            self._adaptive_last_action = "at safe concurrency ceiling"
            return
        self._adaptive_probe_from = self.parallelism
        self._adaptive_probe_rate = rate
        self._set_parallelism_locked(
            min(self.max_parallelism, self.parallelism + self.adaptive_step),
            f"probing above {rate:.1f} tok/s baseline",
        )

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

        # A new block template resets the VDF asynchronously. Launching work
        # before its first checkpoint is guaranteed to fail, yet it is a normal
        # warm-up state rather than a failed vLLM request. Preserve the current
        # adaptive target and let controller polling refill it once ready.
        if not getattr(self.context.read(), "vdf_proof", ""):
            state.waiting_for_vdf = True
            return
        state.waiting_for_vdf = False

        buffered = len(state.results)
        if state.backpressured:
            # A complete inference cohort can finish while the pool is settling already
            # produced shares.  Waiting from max=128 all the way down to a
            # user low-water mark of 32 makes a deterministic multi-second
            # GPU idle valley.  Resume once there is one complete inference
            # batch of verified queue space, while retaining the hard cap.
            resume_at = max(
                self.min_buffered_proofs,
                self.max_buffered_proofs - self.scheduler_parallelism,
            )
            if buffered > resume_at:
                return
            state.backpressured = False
            logger.info(
                "NOMP work %s resumed below proof buffer resume mark (%d)",
                job_id,
                resume_at,
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
        while len(tasks) < self.scheduler_parallelism:
            task = asyncio.create_task(
                self._mine_once(job_id, self._next_admission_delay_locked(state))
            )
            tasks.add(task)
            task.add_done_callback(
                lambda finished, work_id=job_id: self._on_mine_task_done(work_id, finished)
            )

    async def _mine_once(self, job_id: str, admission_delay_seconds: float) -> None:
        """Run exactly one genuine vLLM request for a live NOMP lease."""
        if admission_delay_seconds > 0:
            await asyncio.sleep(admission_delay_seconds)
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
            if not getattr(self.context.read(), "vdf_proof", ""):
                state.waiting_for_vdf = True
                return
            state.waiting_for_vdf = False
            model_name = state.request.model.name
        try:
            await self.request_manager.generate_nomp_dummy(model_name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if VDF_NOT_READY_MESSAGE in str(exc):
                # A block can arrive between the locked readiness check and the
                # proxy's context read. This is expected; wait for the first
                # new VDF checkpoint instead of counting a request failure.
                with self._lock:
                    state = self._jobs_by_id.get(job_id)
                    if state and not state.cancelled:
                        state.waiting_for_vdf = True
                return
            # Avoid a tight error loop if vLLM is restarting.  The done
            # callback refills the slot after this bounded retry delay.
            with self._lock:
                self._mining_errors_total += 1
                self._last_mining_error = str(exc)[:512]
                self._last_mining_error_unix_ms = time.time_ns() // 1_000_000
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
            self._maybe_adjust_parallelism_locked()
            self._ensure_mining_locked(job_id)

    def _schedule_refill_from_callback(self, job_id: str) -> None:
        with self._lock:
            self._ensure_mining_locked(job_id)

    async def submit(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="missing or invalid sidecar token")
        try:
            mine, expires_at = build_nomp_mine_request(
                # Preserve the full safe local scheduling range in request
                # metadata. `_ensure_mining_locked` still admits only the
                # current adaptive target, which can rise after this work unit
                # has already been issued.
                await request.json(), max_parallel=self.max_parallelism
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
            # The Rust controller polls status continuously. It is the fast
            # readiness signal after a fresh VDF checkpoint becomes available.
            self._ensure_mining_locked(job_id)
            task_count = self._active_task_count_locked(job_id)
            common = {
                "ok": True,
                "job_id": job_id,
                "parallelism": self.parallelism,
                "prefetch_requests": self.prefetch_requests,
                "scheduler_target": self.scheduler_parallelism,
                "admission_spread_ms": self.admission_spread_ms,
                "inflight": task_count,
                "buffered_proofs": len(state.results),
                "buffer_limit": self.max_buffered_proofs,
                "waiting_for_vdf": state.waiting_for_vdf,
            }
            if not state.results:
                return web.json_response({**common, "status": "mining"})
            return web.json_response({**common, "status": "proof", **state.results[0]})

    async def metrics(self, request: web.Request) -> web.Response:
        """Expose the rolling vLLM generation rate as a miner performance metric.

        This uses successful completion-token accounting already maintained by
        the local proxy.  It is intentionally distinct from share acceptance:
        share targets influence payout accounting, while tokens/s describes the
        GPU's stable inference throughput for the active model/profile.
        """
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="missing or invalid sidecar token")
        status = self.request_manager.get_status()
        throughput = status.get("throughput", {}) if isinstance(status, dict) else {}
        with self._lock:
            self._prune_expired()
            self._maybe_adjust_parallelism_locked(status)
            for job_id in tuple(self._jobs_by_id):
                self._ensure_mining_locked(job_id)
            active_jobs = list(self._jobs_by_id.items())
            scheduler_inflight = sum(
                self._active_task_count_locked(job_id) for job_id, _ in active_jobs
            )
            buffered_proofs = sum(len(state.results) for _, state in active_jobs)
            duplicate_proofs_dropped = sum(
                state.duplicate_proofs_dropped for _, state in active_jobs
            )
            scheduler_backpressured = any(
                state.backpressured for _, state in active_jobs
            )
            waiting_for_vdf = any(
                state.waiting_for_vdf for _, state in active_jobs
            )
        # This is the proxy's outstanding HTTP request count. The vLLM engine
        # can have a subset Running and the prefetch reserve Waiting; its
        # authoritative Running/Waiting split remains vLLM's own log/metrics.
        active_proxy_requests = int(status.get("active_requests", 0) or 0)
        # In the normal one-job case, a deficit means the sidecar did not
        # create a request.  The queued-or-admitting value is expected to be
        # the prefetch reserve while vLLM is running at full parallelism.
        target_inflight = self.scheduler_parallelism if active_jobs else 0
        return web.json_response(
            {
                "ok": True,
                "generation_tokens_per_sec": float(
                    throughput.get("completion_tokens_per_sec", 0.0) or 0.0
                ),
                "generation_work_units_per_sec": float(
                    throughput.get("hashes_per_sec", 0.0) or 0.0
                ),
                "window_seconds": float(throughput.get("window_seconds", 0.0) or 0.0),
                "configured_parallelism": self.parallelism,
                "adaptive_concurrency": {
                    "enabled": self.adaptive_enabled,
                    "current": self.parallelism,
                    "minimum": self.adaptive_min_parallelism,
                    "maximum": self.max_parallelism,
                    "step": self.adaptive_step,
                    "interval_seconds": self.adaptive_interval_seconds,
                    "last_action": self._adaptive_last_action,
                    "request_errors_total": self._mining_errors_total,
                    "last_request_error": self._last_mining_error,
                    "last_request_error_unix_ms": self._last_mining_error_unix_ms,
                },
                "prefetch_requests": self.prefetch_requests,
                "scheduler_target": target_inflight,
                "scheduler_inflight": scheduler_inflight,
                "scheduler_deficit": max(0, target_inflight - scheduler_inflight),
                "active_proxy_requests": active_proxy_requests,
                # Compatibility with controller v5 and earlier. This has
                # always represented proxy-owned outstanding HTTP requests,
                # not the vLLM engine's Running sequence count.
                "active_requests": active_proxy_requests,
                "request_submission_gap": max(
                    0, scheduler_inflight - active_proxy_requests
                ),
                "buffered_proofs": buffered_proofs,
                "duplicate_proofs_dropped": duplicate_proofs_dropped,
                "scheduler_backpressured": scheduler_backpressured,
                "waiting_for_vdf": waiting_for_vdf,
                "admission_spread_ms": self.admission_spread_ms,
            }
        )

    async def claim(self, request: web.Request) -> web.Response:
        """Lease a bounded batch of queued proofs to one local controller.

        The proof bytes remain in the sidecar until an idempotent acknowledgement
        arrives.  This permits parallel Stratum submission without losing a
        proof if the miner process or its pool connection restarts.
        """
        if not self._authorized(request):
            raise web.HTTPUnauthorized(text="missing or invalid sidecar token")
        try:
            payload = await request.json()
        except (ValueError, TypeError):
            payload = {}
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="claim payload must be a JSON object")
        try:
            limit = _positive_int(payload.get("limit", 1), "limit")
        except MiningProtocolError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        limit = min(limit, 64)
        job_id = request.match_info["job_id"]
        now = time.monotonic()
        with self._lock:
            self._prune_expired()
            state = self._jobs_by_id.get(job_id)
            if not state:
                return web.json_response(
                    {
                        "ok": False,
                        "job_id": job_id,
                        "status": "expired",
                        "error": "unknown, expired, or cancelled work unit",
                    },
                    status=404,
                )
            # Claim polling also serves as a readiness wake-up in controllers
            # that do not call the single-proof status endpoint.
            self._ensure_mining_locked(job_id)
            for proof_id, expires_at in list(state.claims.items()):
                if expires_at <= now:
                    state.claims.pop(proof_id, None)
            claimed: list[dict[str, Any]] = []
            for result in state.results:
                proof_id = int(result["proof_id"])
                if proof_id in state.claims:
                    continue
                state.claims[proof_id] = now + self.claim_lease_seconds
                claimed.append(dict(result))
                # A block candidate is template-changing work. Claim it first
                # and let the controller settle it before leasing older shares.
                if result.get("is_block") or len(claimed) >= limit:
                    break
            common = {
                "ok": True,
                "job_id": job_id,
                "parallelism": self.parallelism,
                "prefetch_requests": self.prefetch_requests,
                "scheduler_target": self.scheduler_parallelism,
                "admission_spread_ms": self.admission_spread_ms,
                "inflight": self._active_task_count_locked(job_id),
                "buffered_proofs": len(state.results),
                "buffer_limit": self.max_buffered_proofs,
                "claim_lease_seconds": self.claim_lease_seconds,
                "waiting_for_vdf": state.waiting_for_vdf,
            }
            if not claimed:
                return web.json_response({**common, "status": "mining", "proofs": []})
            return web.json_response({**common, "status": "proofs", "proofs": claimed})

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
            try:
                payload = await request.json()
            except (ValueError, TypeError):
                payload = {}
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                raise web.HTTPBadRequest(text="acknowledgement payload must be a JSON object")
            proof_ids = payload.get("proof_ids")
            if proof_ids is None:
                # Legacy controllers acknowledge the queue head one at a time.
                acknowledged = state.results.popleft() if state.results else None
                if acknowledged:
                    state.claims.pop(int(acknowledged.get("proof_id", -1)), None)
                acknowledged_results = [acknowledged] if acknowledged else []
            else:
                if not isinstance(proof_ids, list) or len(proof_ids) > 64:
                    raise web.HTTPBadRequest(text="proof_ids must be an array of at most 64 proof ids")
                try:
                    acknowledged_ids = {_positive_int(value, "proof_id") for value in proof_ids}
                except MiningProtocolError as exc:
                    raise web.HTTPBadRequest(text=str(exc)) from exc
                acknowledged_results = [
                    result for result in state.results if int(result.get("proof_id", -1)) in acknowledged_ids
                ]
                if acknowledged_results:
                    state.results = collections.deque(
                        result
                        for result in state.results
                        if int(result.get("proof_id", -1)) not in acknowledged_ids
                    )
                for proof_id in acknowledged_ids:
                    state.claims.pop(proof_id, None)
                acknowledged = acknowledged_results[0] if acknowledged_results else None
            # A block candidate changes the chain template as soon as it is
            # accepted.  Keep any already-buffered proofs visible for the
            # controller to settle, but never start another old-template run.
            if any(result.get("is_block") for result in acknowledged_results):
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
        return web.json_response({"ok": True, "acknowledged": len(acknowledged_results)})

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
            # Do this before allocating a proof_id.  The fingerprint covers
            # the complete FlatBuffer, matching the pool's duplicate identity
            # while avoiding retention of up to megabytes of proof data per
            # completed inference request.
            proof_fingerprint = hashlib.sha256(proof).digest()
            if proof_fingerprint in state.proof_fingerprints:
                state.duplicate_proofs_dropped += 1
                logger.debug(
                    "NOMP sidecar dropped duplicate proof for %s (request %s)",
                    job_id,
                    req_id,
                )
                return
            state.proof_fingerprints.add(proof_fingerprint)
            result = {
                "proof_id": state.next_proof_id,
                "proof_b64": base64.b64encode(proof).decode("ascii"),
                "nonce": str(nonce),
                "achieved_hash": achieved_hash,
                "model_identifier": _extract_model_identifier(proof) or "",
                "is_block": bool(_extract_is_solution(proof)),
                # A diagnostic timestamp only.  The Rust controller uses it
                # to measure proof-ready-to-submit delay; it is never part of
                # the proof bytes or the pool submission contract.
                "produced_at_unix_ms": time.time_ns() // 1_000_000,
            }
            state.next_proof_id += 1
            if result["is_block"]:
                # Block propagation beats ordinary share accounting.  A block
                # is never silently dropped behind a buffered share.
                state.block_pending = True
                state.results.appendleft(result)
                self._cancel_mining_tasks_locked(job_id)
            else:
                state.results.append(result)
                if len(state.results) > self.max_buffered_proofs + self.scheduler_parallelism:
                    logger.error(
                        "NOMP proof buffer overshot its bound for %s: buffered=%d max=%d inflight=%d",
                        job_id,
                        len(state.results),
                        self.max_buffered_proofs,
                        self.scheduler_parallelism,
                    )
            loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._schedule_refill_from_callback, job_id)
