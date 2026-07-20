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
bash start.sh --pool pool.example.org:3336 --wallet 'YOUR_PAYOUT_ADDRESS' --worker 'rig-01'
```

`auto` is the default grouping policy. It creates only valid Tensor Parallel
groups for Qwen3-8B: TP=1, TP=2, or TP=4. You can override it when needed:

```bash
bash start.sh --pool pool.example.org:3336 --wallet 'YOUR_PAYOUT_ADDRESS' --worker 'rig-01' --gpu-groups '0,1;2,3'
```

### One to eight GPUs

The model cannot use TP=3, 5, 6, or 7. The automatic planner uses VRAM, not
just card count:

| Per-card VRAM | Automatic group | 1 / 2 / 3 / 5 / 6 / 7 / 8 cards |
| --- | --- | --- |
| >=22 GiB | TP=1 per card | Every card becomes an independent miner group. |
| 11–21 GiB | TP=2 pairs | Uses 2, 4, 6, or 8 cards; an odd last card waits idle. |
| 7.5–10.9 GiB | TP=4 quartets | Needs 4 cards per group; 8 cards create two groups. |
| <7.5 GiB | Unsupported | The mainnet 8B profile cannot start safely. |

For example, three 12 GB cards become `0,1` with GPU 2 idle; five become
`0,1;2,3` with GPU 4 idle; six become `0,1;2,3;4,5`; and eight 8 GB cards
become `0,1,2,3;4,5,6,7`. The auto planner prints every unused card rather
than silently creating an invalid TP group.

The first launch pulls the public runtime image and downloads the chain-pinned
Qwen3-8B snapshot once. Image pulls and model downloads retry automatically;
completed Docker layers and Hugging Face cache chunks are reused after an
interruption. It writes the model to
`runtime/models` by default. Every sidecar group on that physical host mounts
the same directory, so four GPUs or two groups still use one 16 GB disk cache.
Docker similarly stores the runtime image layers once per host.

The launcher writes a local completion marker only after the entire pinned
snapshot finishes. A partial `config.json` alone is never treated as a complete
model, so rerunning `bash start.sh` resumes the missing weight shards before
vLLM starts in offline mode.

### Slow or interrupted downloads

For mainland-China networks, configuring only `HTTP_PROXY` in the shell does
**not** proxy `docker pull`: Docker pulls are made by the daemon. Configure the
daemon proxy once (this restarts Docker, so do it before starting miners):

```bash
bash docker-proxy.sh --proxy http://PROXY_HOST:PORT
```

For the Hugging Face model download, either export the standard proxy variables
before starting, or add `TENSORCASH_HTTP_PROXY=http://PROXY_HOST:PORT` to the
host-local `miner.env`. The launcher forwards it into the downloader container.

If a registry route remains unreliable, create a source-free seed bundle on one
completed host and move it using `rsync`; unlike `scp`, it continues a partial
14 GB image/model transfer and verifies appended data:

```bash
bash seed-export.sh --copy-to root@DESTINATION_HOST:/root/
```

For an HTTP/object-storage mirror, set these host-local values before the first
start. `curl --continue-at -` retains `<archive>.partial` and resumes it on the
next `bash start.sh` invocation:

```bash
TENSORCASH_IMAGE_ARCHIVE_URL='https://mirror.example/tensorcash-image.tar.zst'
TENSORCASH_IMAGE_ARCHIVE_SHA256='64_HEX_SHA256_OF_THE_ARCHIVE'
```

## Seed one host, then start other hosts with one command

The seed script packages the loaded runtime image, the pinned model cache, and
the public launcher scripts. It deliberately excludes `miner.env` and
`runtime/data`, so no payout address, sidecar token, or machine-local proof
state is copied. It creates checksums, then optionally transfers the complete
bundle through `scp`.

On the fully downloaded seed host:

```bash
cd ~/tensorcash-miner
git pull --ff-only
bash seed-export.sh --copy-to root@DESTINATION_HOST:/root/
```

On the destination host, run exactly one command after the transfer completes:

```bash
bash /root/tensorcash-seed-mainnet-0.1.0/seed-install.sh
```

The installer verifies every file, installs `zstd` automatically on Ubuntu if
needed, loads the local Docker image, extracts the local model cache, prompts
for the payout account, and starts the GPU groups. It uses the pool endpoint
from the seed host by default. Override values for non-interactive deployment:

```bash
bash /root/tensorcash-seed-mainnet-0.1.0/seed-install.sh \
  --pool pool.example.org:3336 \
  --wallet 'YOUR_PAYOUT_ADDRESS' \
  --worker 'rig-01' \
  --gpu-groups auto
```

For many machines in one LAN, copy the generated seed directory to a local
NAS/object store and then run its `seed-install.sh` on each machine. Extract
the model to each host's local NVMe; do not run vLLM directly from a shared
network filesystem.

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

# Follow only the controller. It reports rolling PoI/s, 5-minute proofs and
# shares, accepted/rejected totals, proof latency, and last-valid time every
# 30 seconds by default.
docker logs -f tensorcash-rig-01-g1-miner-1

# Stop all groups created by this launcher.
bash start.sh --stop
```

Set `TENSORCASH_STATS_INTERVAL=30` in the host-local `miner.env` to change the
report cadence; set it to `0` to disable periodic controller statistics. PoI/s
is accepted proof-of-inference shares per second, not a SHA hash rate.

## Bounded inference concurrency

The default remains one sequence per group for compatibility. The sidecar's
NOMP scheduler can keep several independent, canonical inference attempts for
the same live work unit in flight. It has a bounded proof queue and prioritizes
a block candidate over ordinary shares; it does not change the registered
model, header, target, VDF, or proof verification rules.

For an RTX 4090 with the 8B profile, test four slots first by adding these
host-local values to `miner.env`, then restarting the launcher:

```bash
VLLM_MAX_NUM_SEQS=4
NOMP_SIDECAR_CONCURRENCY=4
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=2
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=8
```

Keep `NOMP_SIDECAR_CONCURRENCY` less than or equal to
`VLLM_MAX_NUM_SEQS`. Test one profile for at least ten minutes against the
real pool and verifier before increasing it. Start 12 GB TP=2 groups at two
slots; do not enable four or eight slots by default on 8 GB configurations.
The useful metric is sustained submitted/accepted PoI rate with zero stale or
invalid proofs, not a momentary GPU power draw.

## Updating without re-downloading the model

```bash
bash start.sh --update
```

The update command pulls `mainnet-latest`, records its immutable digest in the
host-local `miner.env`, and recreates the sidecar/miner containers. Docker
reuses the existing CUDA/vLLM base layers; the shared `runtime/models` mount is
not removed or downloaded again. A normal miner update must not change the
chain-pinned model profile. A model/profile migration is a separately announced
pool upgrade and requires explicit operator action.

The image omits source code but no client-side binary is impossible to reverse
engineer. Treat this as source-distribution control, not as a cryptographic IP
boundary.

## Architecture support

The Rust controller is CPU-only and therefore has no CUDA `sm_*` fatbins: the
same `linux/amd64` binary controls every supported GPU. The launcher downloads
a pinned, SHA-256-verified controller release asset (about 3 MB) built for
glibc 2.34, then mounts it over the runtime image's controller. This keeps
Ubuntu 22.04 and HiveOS Docker hosts compatible without downloading a second
large runtime image. Actual inference runs in the CUDA/vLLM sidecar, so GPU
compatibility is determined by that pinned runtime and available VRAM, not by
the Rust binary. RTX 4070 Super has been tested; ARM64 and untested GPU
generations are not advertised as supported.
