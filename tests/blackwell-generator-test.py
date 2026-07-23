#!/usr/bin/env python3
"""Regression checks for the generated Blackwell compatibility stage."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "blackwell" / "generate_dockerfile.py"


def main() -> None:
    # The generator deliberately validates these two upstream blocks before
    # extending the recipe.  A minimal source with both blocks exercises the
    # actual command-line generation path without attempting a CUDA build.
    source = """FROM nvcr.io/nvidia/pytorch:26.03-py3
ARG CUDA_VERSION

ENV DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST=\"12.0;12.0+PTX\" \\
    VLLM_TARGET_DEVICE=cuda \\
    MAX_JOBS=4 \\
    CCACHE_DIR=/ccache \\
    VLLM_INSTALL_PUNICA_KERNELS=0 \\
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.19.0+pow
"""

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        upstream = temporary / "upstream.Dockerfile"
        output = temporary / "generated.Dockerfile"
        upstream.write_text(source, encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--upstream",
                str(upstream),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rendered = output.read_text(encoding="utf-8")

    assert 'if ! getent passwd 1000 >/dev/null; then useradd -m -u 1000 vllm; fi;' in rendered
    assert 'runtime_user="$(getent passwd 1000 | cut -d: -f1)";' in rendered
    assert 'user=$runtime_user' in rendered
    assert 'chown -R "$runtime_user:$runtime_group"' in rendered
    assert "chown -R vllm:vllm" not in rendered
    print("blackwell generator tests: OK")


if __name__ == "__main__":
    main()
