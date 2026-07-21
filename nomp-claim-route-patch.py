#!/usr/bin/env python3
"""Idempotently add the NOMP proof-claim route to the pinned proxy runtime."""

from __future__ import annotations

from pathlib import Path


PATH = Path("/app/miner-proxy/src/main.py")
MARKER = "app.router.add_post('/v1/tensorcash/jobs/{job_id}/claims', self.nomp_sidecar.claim)"
NEEDLE = "app.router.add_get('/v1/tensorcash/jobs/{job_id}', self.nomp_sidecar.status)"


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if MARKER in text:
        print("[TensorCash] NOMP proof-claim route already present")
        return
    if text.count(NEEDLE) != 1:
        raise SystemExit(
            "Pinned miner proxy has an unexpected NOMP route layout; refusing an unsafe patch"
        )
    indent = "            "
    insertion = f"{NEEDLE}\n{indent}{MARKER}"
    PATH.write_text(text.replace(NEEDLE, insertion), encoding="utf-8")
    print("[TensorCash] Installed NOMP proof-claim route")


if __name__ == "__main__":
    main()
