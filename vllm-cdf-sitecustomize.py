"""TensorCash numerical safety hook loaded by the vLLM Python interpreter.

The pinned runtime image may start vLLM as an unprivileged user, so modifying
site-packages in place is not reliable.  Python imports ``sitecustomize`` before
the vLLM CLI, allowing this read-only overlay to canonicalize the categorical
CDF terminal boundary without changing the image or registered model.
"""

from __future__ import annotations

from functools import wraps

from vllm.sampling.pow_utils import PowHasher


_MARKER = "_tensorcash_closed_cdf_boundary"


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


_install()
