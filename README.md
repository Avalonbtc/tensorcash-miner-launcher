# TensorCash miner launcher

This is the public, source-free launcher for the TensorCash miner. It contains
only Docker Compose and shell scripts. The private build pipeline publishes a
stripped miner binary inside `ghcr.io/avalonbtc/tensorcash-miner`; no Rust
checkout, wallet, pool configuration, or model cache is published here.

## Start

On a `linux/amd64` Linux/Vast host with Docker Compose v2, NVIDIA Container
Toolkit, and GPUs visible through `nvidia-smi`:

```bash
git clone https://github.com/Avalonbtc/tensorcash-miner-launcher.git ~/tensorcash-miner
cd ~/tensorcash-miner
bash start.sh --pool pool.example.org:3336 --wallet 'YOUR_PAYOUT_ADDRESS' --worker 'rig-01' --gpu-groups '0,1,2,3'
```

For two independent two-GPU proof streams on one four-GPU host:

```bash
bash start.sh --pool pool.example.org:3336 --wallet 'YOUR_PAYOUT_ADDRESS' --worker 'rig-01' --gpu-groups '0,1;2,3'
```

The first launch pulls the public runtime image and downloads the chain-pinned
Qwen3-8B snapshot once. It shows download progress and writes it to
`runtime/models` by default. Every sidecar group on that physical host mounts
the same directory, so four GPUs or two groups still use one 16 GB disk cache.
Docker similarly stores the runtime image layers once per host.

## What can and cannot be reused

- Same host: the image layers and model cache are shared automatically; never
  create one model directory per GPU group.
- Multiple independent rental machines: each physical host must have access to
  the model weights at least once. Use a provider custom template/snapshot or a
  regional registry/object-storage mirror to seed it quickly; a model needed
  for local inference cannot be made to occupy zero bytes on a new host.
- Upgrades: the 14 GB CUDA/vLLM base is a stable Docker layer. Releases add a
  small binary overlay, so existing hosts fetch only changed layers.

`miner.env` is created with mode `0600` and must remain local. It holds the
payout account and a host-local sidecar token. Do not expose Docker, sidecar,
or pool infrastructure ports to the Internet.

## Operations

```bash
# Show all groups and their health.
docker ps --filter 'name=tensorcash-'

# Follow a specific group, for example group 1.
docker compose -p tensorcash-rig-01-g1 --env-file miner.env logs -f

# Stop all groups created by this launcher.
bash start.sh --stop
```

The image omits source code but no client-side binary is impossible to reverse
engineer. Treat this as source-distribution control, not as a cryptographic IP
boundary.

## Architecture support

The Rust controller is CPU-only and therefore has no CUDA `sm_*` fatbins: the
same `linux/amd64` binary controls every supported GPU. Actual inference runs
in the CUDA/vLLM sidecar, so GPU compatibility is determined by that pinned
runtime and available VRAM, not by the Rust binary. RTX 4070 Super has been
tested; ARM64 and untested GPU generations are not advertised as supported.
