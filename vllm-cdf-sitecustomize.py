"""TensorCash numerical safety hook loaded by the vLLM Python interpreter.

The pinned runtime image may start vLLM as an unprivileged user, so modifying
site-packages in place is not reliable.  Python imports ``sitecustomize`` before
the vLLM CLI, allowing this read-only overlay to canonicalize the categorical
CDF terminal boundary without changing the image or registered model.
"""

from __future__ import annotations

import os
from functools import wraps

from vllm.sampling.pow_utils import PowHasher


_MARKER = "_tensorcash_closed_cdf_boundary"
_ROWS_MARKER = "_tensorcash_configured_row_capacity"
_ROWS_EVICTION_MARKER = "_tensorcash_batch_protected_row_eviction"


def _install() -> None:
    original = PowHasher.batch_sample_tokens
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def batch_sample_tokens_with_closed_tail(self, contexts, steps, cdfs, *args, **kwargs):
        # SHA-derived u is in [0, 1).  Float32 cumsum can land a few ulps below
        # one, which makes searchsorted return V and later trips a CUDA gather
        # assertion. A categorical CDF is canonically closed at its last token.
        if cdfs.ndim != 2 or cdfs.size(1) == 0:
            raise ValueError("batch_sample_tokens requires a non-empty [B, V] CDF")
        cdfs[..., -1] = 1.0
        return original(self, contexts, steps, cdfs, *args, **kwargs)

    setattr(batch_sample_tokens_with_closed_tail, _MARKER, True)
    PowHasher.batch_sample_tokens = batch_sample_tokens_with_closed_tail

    # TensorCash's PoW sampler originally allocated exactly 1024 bookkeeping
    # rows. vLLM can sample both running requests and its bounded wait reserve
    # in one pass, so a 1024-running + 64-prefetch profile could evict rows
    # from its own batch and crash EngineCore. Docker images are patched at
    # import time because their pinned wheel is intentionally read-only.
    from vllm.v1.sample.ops.topk_topp_sampler import PowTopKTopPSampler

    original_init = PowTopKTopPSampler.__init__
    if getattr(original_init, _ROWS_MARKER, False):
        return

    @wraps(original_init)
    def init_with_configured_row_capacity(self, *args, **kwargs):
        if not args and "max_concurrency" not in kwargs:
            raw_capacity = os.getenv("POW_MAX_CONCURRENCY", "1024").strip()
            try:
                capacity = int(raw_capacity)
            except ValueError as exc:
                raise RuntimeError(
                    "POW_MAX_CONCURRENCY must be a positive integer") from exc
            if not 1 <= capacity <= 4096:
                raise RuntimeError(
                    "POW_MAX_CONCURRENCY must be between 1 and 4096")
            kwargs["max_concurrency"] = capacity
        return original_init(self, *args, **kwargs)

    setattr(init_with_configured_row_capacity, _ROWS_MARKER, True)
    PowTopKTopPSampler.__init__ = init_with_configured_row_capacity

    # Keep every row that participates in the current vLLM sample batch until
    # the batch has consumed it. The original TensorCash helper could evict an
    # earlier row from the same batch while allocating a late new request,
    # causing get_row() to return None later in the sampler. Native launches
    # receive the equivalent audited source patch; Docker uses this import-time
    # overlay because the pinned wheel remains read-only.
    from vllm.sampling.common_sampler_helper import CommonSamplerHelper

    original_ensure_rows = CommonSamplerHelper.ensure_rows
    if getattr(original_ensure_rows, _ROWS_EVICTION_MARKER, False):
        return

    @wraps(original_ensure_rows)
    def ensure_rows_with_batch_protection(self, seq_ids, prompt_mapping):
        protected = set(seq_ids)
        for sid in seq_ids:
            if sid in self.s.row_manager.seqid_to_row:
                continue
            row = self.s.row_manager.allocate_row(sid)
            if row is None:
                candidates = [
                    candidate for candidate in self.s.row_manager.seqid_to_row
                    if candidate not in protected
                ]
                if candidates:
                    old_sid = min(
                        candidates,
                        key=lambda candidate: self.s.row_manager.allocation_order.get(
                            candidate, float("inf")),
                    )
                    self.s._free_sequence(old_sid)
                    row = self.s.row_manager.allocate_row(sid)
            if row is None:
                raise RuntimeError(
                    "TensorCash PoW row pool cannot cover the current vLLM "
                    f"sample batch ({len(protected)} protected rows; "
                    f"capacity={self.s.row_manager.max_rows}); increase "
                    "POW_MAX_CONCURRENCY")
            self.s.ring_buffers.clear_row(row)
            self.s._init_sequence_cache(sid, prompt_mapping[sid])
            seq_params = self.s.seq_params.get(sid, {})
            pow_snapshot = seq_params.get("pow_snapshot")
            if pow_snapshot:
                self.s.ring_buffers.write_pow_params(row, pow_snapshot)

    setattr(ensure_rows_with_batch_protection, _ROWS_EVICTION_MARKER, True)
    CommonSamplerHelper.ensure_rows = ensure_rows_with_batch_protection


_install()
