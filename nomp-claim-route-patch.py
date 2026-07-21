#!/usr/bin/env python3
"""Idempotently add NOMP claim and performance-metric routes to the runtime."""

from __future__ import annotations

from pathlib import Path


PATH = Path("/app/miner-proxy/src/main.py")
CLAIM_MARKER = "app.router.add_post('/v1/tensorcash/jobs/{job_id}/claims', self.nomp_sidecar.claim)"
METRICS_MARKER = "app.router.add_get('/v1/tensorcash/metrics', self.nomp_sidecar.metrics)"
NEEDLE = "app.router.add_get('/v1/tensorcash/jobs/{job_id}', self.nomp_sidecar.status)"


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    missing = []
    if CLAIM_MARKER not in text:
        missing.append(CLAIM_MARKER)
    if METRICS_MARKER not in text:
        missing.append(METRICS_MARKER)
    if not missing:
        print("[TensorCash] NOMP claim and metrics routes already present")
        return
    if text.count(NEEDLE) != 1:
        raise SystemExit(
            "Pinned miner proxy has an unexpected NOMP route layout; refusing an unsafe patch"
        )
    indent = "            "
    insertion = f"{NEEDLE}\n" + "\n".join(f"{indent}{route}" for route in missing)
    PATH.write_text(text.replace(NEEDLE, insertion), encoding="utf-8")
    print("[TensorCash] Installed NOMP claim/metrics route overlay")


if __name__ == "__main__":
    main()
