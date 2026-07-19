# TensorCash Vast.ai image launcher

This repository contains only Docker Compose configuration and launch scripts.
It contains no Rust miner source, TensorCash source tree, wallet, pool runtime
configuration, model cache, or private sidecar token. The runtime pulls two
prebuilt GHCR images:

- `ghcr.io/avalonbtc/tensortest-sidecar:0.1.0` — TensorCash vLLM proof sidecar
- `ghcr.io/avalonbtc/tensortest-miner:0.1.0` — compiled Rust controller only

## Start on Vast.ai

Use one instance containing four GPUs on the same physical host. Four RTX 4070
Super cards normally provide 12 GB each; the test image uses TP=4 with GPUs
`0,1,2,3`.

```bash
git clone https://github.com/Avalonbtc/tensortest.git ~/tensortest
cd ~/tensortest
PAYOUT_ACCOUNT='replace-with-your-payout-address' WORKER='vast-4070s-01' bash start.sh
```

The first command generates a private local `miner.env`, validates Docker GPU
visibility, pulls the images, and starts the two containers. The model cache is
persisted under `~/tensortest/runtime`. Use `Ctrl+C` only to stop following
logs; the miner containers remain running.

To inspect or stop the miner:

```bash
docker compose --env-file miner.env logs -f miner
docker compose --env-file miner.env down
```

## Scope and security

The active pool is a TCP-only test endpoint at `119.91.239.215:3336` and uses
the registered `Qwen/Qwen3-0.6B` test profile. It validates four-GPU inference,
proof generation, and NOMP submission; it is not an 8B production pool.

The images do not contain the Rust source tree, but any distributed binary can
still be reverse engineered. Do not treat a public container registry as a
source-code confidentiality boundary. Do not expose the sidecar, its token,
Docker socket, BCore RPC, or verifier ports to the public Internet.
