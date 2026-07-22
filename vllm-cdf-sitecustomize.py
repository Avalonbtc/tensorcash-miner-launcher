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


_install()
