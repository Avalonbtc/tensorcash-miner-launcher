#!/usr/bin/env python3
"""Generate the TensorCash Blackwell simple-worker runtime Dockerfile.

The upstream repository already owns the difficult part of the image: a
vLLM 0.19 wheel compiled against NVIDIA's CUDA-13 / Blackwell PyTorch stack
with TensorCash' PoW sampler.  The public miner launcher needs the same
runtime to also contain its local miner-proxy and supervisord programs.

Keeping the extension in this small generator avoids copying and silently
diverging from the upstream Blackwell build recipe.
"""

from __future__ import annotations

import argparse
from pathlib import Path


UPSTREAM_RUNTIME_STAGE = """FROM nvcr.io/nvidia/pytorch:26.03-py3
ARG CUDA_VERSION

ENV DEBIAN_FRONTEND=noninteractive
"""

RUNTIME_STAGE = """FROM nvcr.io/nvidia/pytorch:26.03-py3 AS blackwell-runtime
ARG CUDA_VERSION

ENV DEBIAN_FRONTEND=noninteractive
"""

VLLM_BUILD_ENV = """ENV TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX" \\
    VLLM_TARGET_DEVICE=cuda \\
    MAX_JOBS=4 \\
    CCACHE_DIR=/ccache \\
    VLLM_INSTALL_PUNICA_KERNELS=0 \\
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.19.0+pow
"""

VLLM_BUILD_ENV_THROTTLED = """# GitHub-hosted runners have finite RAM, but a single CUDA compile
# job exceeds their six-hour wall-clock limit. Build only Blackwell SASS and
# use two compiler jobs while BuildKit keeps independent heavy stages serial.
ARG VLLM_BUILD_JOBS=2
ENV TORCH_CUDA_ARCH_LIST="12.0" \\
    VLLM_TARGET_DEVICE=cuda \\
    MAX_JOBS=${VLLM_BUILD_JOBS} \\
    CMAKE_BUILD_PARALLEL_LEVEL=${VLLM_BUILD_JOBS} \\
    CCACHE_DIR=/ccache \\
    VLLM_INSTALL_PUNICA_KERNELS=0 \\
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.19.0+pow
"""


SIMPLE_WORKER_STAGE = r'''

# =============================================================================
# TensorCash NOMP simple-worker compatibility layer
#
# This stage intentionally retains the launcher-facing layout from
# mainnet-0.1.0.  The public launcher mounts its audited scheduler and startup
# overlays at these paths, so miners can move to Blackwell without a protocol
# or configuration change.
# =============================================================================
FROM blackwell-runtime AS tensorcash-miner-blackwell

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl supervisor && \
    rm -rf /var/lib/apt/lists/*

# The Blackwell vLLM runtime needs NumPy 2.  The legacy proxy requirements
# pins NumPy 1.26 for vLLM 0.10, so install every proxy dependency except that
# obsolete pin rather than letting pip downgrade the working CUDA runtime.
COPY services/miner-api/proxy_requirements.txt /tmp/proxy_requirements.txt
RUN sed -i -E '/^numpy==/d' /tmp/proxy_requirements.txt && \
    pip install --no-cache-dir -r /tmp/proxy_requirements.txt && \
    rm -f /tmp/proxy_requirements.txt

# Miner proxy and TensorCash proof helpers.  The launcher mounts the current
# NOMP controller module at runtime, but the rest of the proxy remains image
# owned and versioned with this source revision.
COPY services/miner-api/src /app/miner-proxy/src
COPY shared-utils/pow-utils/pow_utils.py /app/miner-proxy/src/utils/
COPY shared-utils/pow-utils/pow_v3.py /app/miner-proxy/src/utils/
COPY shared-utils/pow-utils/bcred_table_r1024.py /app/miner-proxy/src/utils/
COPY shared-utils/pow-utils/uint256_arithmetics.py /app/miner-proxy/src/utils/
COPY shared-utils/config/constants.py /app/miner-proxy/src/config/
RUN mkdir -p /app/miner-proxy/src/proof && \
    cp -r /app/proof/* /app/miner-proxy/src/proof/

COPY deployments/simple-worker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deployments/simple-worker/start-vllm.sh /app/start-vllm.sh
COPY deployments/simple-worker/start-proxy.sh /app/start-proxy.sh
COPY deployments/simple-worker/start-vllm-mining.sh /app/start-vllm-mining.sh
COPY deployments/simple-worker/chat-template/qwen3.5-enhanced.jinja /opt/chat-template/qwen3.5-enhanced.jinja

# The upstream Blackwell stage deliberately runs as numeric uid 1000, but the
# NVIDIA base image does not promise a particular *name* for that account.  In
# particular, it need not be `vllm`; assuming that name made the final image
# layer fail only after the expensive CUDA build had completed.  Resolve the
# existing uid-1000 account at build time (creating the conventional account
# only if the base image exposes uid 1000 numerically without a passwd entry)
# and make the legacy supervisor template use that exact user and home
# directory.
RUN set -eu; \
    if ! getent passwd 1000 >/dev/null; then useradd -m -u 1000 vllm; fi; \
    runtime_user="$(getent passwd 1000 | cut -d: -f1)"; \
    runtime_group="$(id -gn "$runtime_user")"; \
    runtime_home="$(getent passwd "$runtime_user" | cut -d: -f6)"; \
    test -n "$runtime_user" && test -n "$runtime_group" && test -n "$runtime_home"; \
    sed -i \
        -e "s/^user=worker$/user=$runtime_user/" \
        -e "s@HOME=\"/home/worker\",USER=\"worker\"@HOME=\"$runtime_home\",USER=\"$runtime_user\"@" \
        /etc/supervisor/conf.d/supervisord.conf; \
    sed -i 's/\r$//' /app/start-vllm.sh /app/start-proxy.sh /app/start-vllm-mining.sh && \
    chmod 0755 /app/start-vllm.sh /app/start-proxy.sh /app/start-vllm-mining.sh && \
    mkdir -p /models /data /opt/tensorcash /var/log/supervisor && \
    chown -R "$runtime_user:$runtime_group" /app /models /data /var/log/supervisor /opt/tensorcash

ENV PYTHONPATH="/app:/app/miner-proxy/src:${PYTHONPATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    MODEL_NAME="Qwen/Qwen3-8B" \
    MODEL_HASH="" \
    MAX_MODEL_LEN=2048 \
    GPU_MEM_UTIL=0.89 \
    VLLM_ENABLE_POW=1 \
    API_KEY="internal-secret" \
    WORKER_MODE=standalone \
    STANDALONE_MODE=true \
    TARGET_URL="http://127.0.0.1:8000" \
    HTTP_HOST="0.0.0.0" \
    HTTP_PORT=8080 \
    PROOF_CACHE_ENABLED=false \
    PROOF_COLLECTOR_PORT=7002 \
    MINING_ENABLED=true

# This is a build-time ABI guard, not a GPU smoke test.  It fails the image
# before publication if the selected NVIDIA PyTorch base ever stops carrying
# Blackwell code objects.
RUN python3 -c 'import chiavdf, proof_processor, torch, vllm; arches = set(torch.cuda.get_arch_list()); assert any(arch.startswith("sm_120") for arch in arches), arches; print("TensorCash Blackwell runtime build check: OK", sorted(arches))'

EXPOSE 7002 8000 8080

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.upstream.read_text(encoding="utf-8")
    if source.count(UPSTREAM_RUNTIME_STAGE) != 1:
        raise SystemExit(
            "Unsupported upstream Blackwell Dockerfile: expected one final "
            "NVIDIA PyTorch runtime stage."
        )
    if source.count(VLLM_BUILD_ENV) != 1:
        raise SystemExit(
            "Unsupported upstream Blackwell Dockerfile: expected one vLLM "
            "source-build environment block."
        )
    rendered = source.replace(UPSTREAM_RUNTIME_STAGE, RUNTIME_STAGE, 1)
    rendered = rendered.replace(VLLM_BUILD_ENV, VLLM_BUILD_ENV_THROTTLED, 1)
    rendered += SIMPLE_WORKER_STAGE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"generated {args.output}")


if __name__ == "__main__":
    main()
