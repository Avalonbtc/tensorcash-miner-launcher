"""Small runtime compatibility patch for the pinned TensorCash sidecar image.

The image returns a plain-text 404 when a local NOMP lease has naturally
expired or was superseded.  That is a normal state transition, but a controller
needs structured data to distinguish it from an HTTP/proxy failure.  Python
loads ``sitecustomize`` before ``main.py`` imports the sidecar controller, so
this keeps existing downloaded images compatible without rebuilding the large
GPU runtime image.
"""

from aiohttp import web

from components.nomp_sidecar import NompSidecarController


_original_status = NompSidecarController.status
_original_acknowledge = NompSidecarController.acknowledge


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


NompSidecarController.status = _structured_status
NompSidecarController.acknowledge = _idempotent_acknowledge
