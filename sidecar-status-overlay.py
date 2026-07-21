"""Small runtime compatibility patch for the pinned TensorCash sidecar image.

The image returns a plain-text 404 when a local NOMP lease has naturally
expired or was superseded.  That is a normal state transition, but a controller
needs structured data to distinguish it from an HTTP/proxy failure.  Python
loads ``sitecustomize`` before ``main.py`` imports the sidecar controller, so
this keeps existing downloaded images compatible without rebuilding the large
GPU runtime image.
"""

import logging
import os

# `sitecustomize` is imported before the sidecar's constants module.  Make the
# continuous-NOMP rule resilient even when an older runtime image or inherited
# host environment supplies a non-zero generic solution cooldown: that setting
# aborts the next vLLM completion after every share and is not consensus work.
os.environ["MINING_SOLUTION_COOLDOWN_SEC"] = "0"

from aiohttp import web

from components.nomp_sidecar import NompSidecarController
from components.vdf_service import VDFService


logger = logging.getLogger(__name__)


_original_status = NompSidecarController.status
_original_claim = NompSidecarController.claim
_original_acknowledge = NompSidecarController.acknowledge
_original_reset_prover = VDFService._reset_prover


def _release_vdf_prover_on_new_block(self, block_hash):
    """Recreate the native VDF prover when its chain challenge changes.

    ``chiavdf.StreamingProver.reset()`` clears its intermediate vectors but
    retains their allocated capacity. Mining has one VDF prover per sidecar,
    so successive blocks otherwise leave each sidecar's Python process at the
    largest checkpoint footprint it has ever reached. A new block invalidates
    the old challenge and all of its checkpoints anyway. Stop and drop the
    native object first so its C++ destructor releases that allocation before
    constructing the prover for the new challenge.
    """
    if self.prover is not None and block_hash != self._current_block_hash:
        old_prover = self.prover
        # Both attributes own a Python reference to the same pybind object.
        # Clear both before dropping the last local reference so destruction is
        # deterministic rather than waiting for a future GC cycle.
        self.prover = None
        self._prover = None
        try:
            old_prover.stop()
        except Exception:  # pragma: no cover - preserve VDF recovery semantics
            logger.exception("Failed to stop superseded TensorCash VDF prover")
        finally:
            del old_prover
        logger.info(
            "Released superseded TensorCash VDF prover before challenge reset"
        )
    return _original_reset_prover(self, block_hash)


async def _structured_status(self, request):
    try:
        return await _original_status(self, request)
    except web.HTTPNotFound:
        return web.json_response(
            {
                "ok": False,
                "job_id": request.match_info.get("job_id", ""),
                "status": "expired",
                "error": "unknown, expired, or cancelled work unit",
            },
            status=404,
        )


async def _idempotent_acknowledge(self, request):
    try:
        return await _original_acknowledge(self, request)
    except web.HTTPNotFound:
        return web.json_response(
            {
                "ok": True,
                "acknowledged": False,
                "status": "expired",
            }
        )


async def _structured_claim(self, request):
    try:
        return await _original_claim(self, request)
    except web.HTTPNotFound:
        return web.json_response(
            {
                "ok": False,
                "job_id": request.match_info.get("job_id", ""),
                "status": "expired",
                "error": "unknown, expired, or cancelled work unit",
            },
            status=404,
        )


NompSidecarController.status = _structured_status
NompSidecarController.claim = _structured_claim
NompSidecarController.acknowledge = _idempotent_acknowledge
VDFService._reset_prover = _release_vdf_prover_on_new_block
