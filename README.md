# TensorCash miner launcher

This is the public, source-free launcher for the TensorCash miner. It contains
only Docker Compose and shell scripts. The private build pipeline publishes a
stripped miner binary inside `ghcr.io/avalonbtc/tensorcash-miner`; no Rust
checkout, wallet, pool configuration, or model cache is published here.

## Choose a launch mode

- **Docker mode:** use `start.sh` on normal Linux/HiveOS hosts with Docker
  Compose v2 and NVIDIA Container Toolkit. It supports TP=1, TP=2, and TP=4
  GPU groups.
- **Native mode:** use `native-vast.sh` only in hosted GPU containers that
  expose `/dev/nvidia*` but intentionally provide no Docker daemon. It uses
  one independent TP=1 instance for every selected >=22 GiB GPU.

## Docker mode

On a `linux/amd64` Linux/Vast host with Docker Compose v2, NVIDIA Container
Toolkit, and GPUs visible through `nvidia-smi`:

```bash
git clone https://github.com/Avalonbtc/tensorcash-miner-launcher.git ~/tensorcash-miner
cd ~/tensorcash-miner
bash start.sh --pool pool.example.org:3336 --wallet 'YOUR_PAYOUT_ADDRESS' --worker 'rig-01'
```

### RTX 50-series / Blackwell

The legacy `mainnet-0.1.0` image contains a PyTorch build ending at `sm_90`.
It cannot execute on an RTX 5090 (`sm_120`).  On a compute-capability 12.x
host, a first launch selects
`ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.1-blackwell`; an existing
config that uses the exact legacy default tag is migrated to that tag at the
next `bash start.sh`. Custom tags and immutable digests are never changed.

To set the image explicitly instead:

```bash
cd ~/tensorcash-miner
sed -i 's|^MINER_IMAGE=.*|MINER_IMAGE=ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.1-blackwell|' miner.env
bash start.sh
```

The Blackwell tag is a separate CUDA 13 / PyTorch / vLLM build. It uses the
same chain-pinned model, proof format, pool protocol, controller, and local
sidecar API as the standard image; it is not a consensus or mining-policy
change.

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

## Docker operation and inspection

```bash
# Show all groups and their health.
docker ps --filter 'name=tensorcash-'

# Follow a specific group, for example group 1.
docker compose -p tensorcash-rig-01-g1 --env-file miner.env logs -f

# Follow only the controller. It reports its explicit rolling window,
# rolling `generation=…tok/s` GPU performance rating, raw accepted shares/s,
# model-difficulty-normalized PoI/s, a separate
# network-target-equivalent estimate, sidecar queue pressure, and phase
# timing for sidecar start/claim/ack, proof delivery, and pool response.
# The report is every 30 seconds by default.
docker logs -f tensorcash-rig-01-g1-miner-1

# Stop all groups created by this launcher.
bash start.sh --stop

# Start existing local configuration again, without re-pulling an already
# loaded runtime image or the existing model cache.
TENSORCASH_SKIP_IMAGE_PULL=true bash start.sh
```

For a sidecar scheduler/launcher update, pull the scripts and recreate the
containers. The model cache and Docker image layers remain local:

```bash
cd ~/tensorcash-miner
git pull --ff-only
bash start.sh --stop
TENSORCASH_SKIP_IMAGE_PULL=true bash start.sh
```

Inspect the authenticated sidecar's local queue and generation metric without
printing its token:

```bash
cd ~/tensorcash-miner
set -a && source miner.env && set +a
CID=tensorcash-rig-01-g1-sidecar-1
docker exec -e NOMP_SIDECAR_TOKEN="$NOMP_SIDECAR_TOKEN" "$CID" sh -lc \
  'curl -fsS -H "Authorization: Bearer $NOMP_SIDECAR_TOKEN" http://127.0.0.1:8080/v1/tensorcash/metrics'
```

Set `TENSORCASH_STATS_INTERVAL=30` in the host-local `miner.env` to change the
report cadence; set it to `0` to disable periodic controller statistics.
`generation=…tok/s` is the rolling completion-token throughput for the active
model and mining profile. For a fixed model, prompt recipe, 256-token proof
length, and vLLM settings, this is the GPU's comparable mining performance
rating (the familiar hashrate-style display). The adjacent `work=…/s` is that
same value divided by 256 generated tokens. `shares/s` is the raw accepted
pool-share rate. `norm-PoI/s` scales accepted
shares by the chain-pinned model difficulty relative to
`MODEL_DIFFICULTY_NORMALIZER`; it is useful for comparing model profiles, not
a SHA hash rate. `network-target-eq/s` is separately labelled because it is an
expected network-target hit rate derived from the current share target, not a
claim that a block has been found. Before the process reaches five minutes,
`window=` shows the actual warm-up duration instead of pretending the count is
already a full five-minute measurement.

## Adaptive inference concurrency

The default is adaptive: every group starts at **32** concurrent requests, and
the sidecar probes one 32-request higher level only after a full 60-second
generation window. It keeps the candidate only when vLLM fills at least 75% of
that target and rolling completion throughput improves by at least 2%; it rolls
back for an underfilled target, a 5% regression, or any local vLLM request
error. Completed proofs are never cancelled merely because the target changes.

The launcher does **not** derive a per-TP or per-VRAM concurrency ceiling. It
probes from 32 until vLLM admission, sustained generation, or a real local
error says to stop. The default engineering circuit breaker is 1024; it only
prevents an accidental unbounded coroutine storm and is not a GPU-tier
performance cap. This is a local scheduling policy, not a model, proof,
target, VDF, or consensus change.

`vllm --max-num-seqs` is an initialization allocation, not a live scheduler
knob. A fresh runtime therefore discovers its bootable vLLM capacity in the
same ascending sequence (`32`, `64`, `128`, ...), stops at the first failed
bootstrap, and records the last healthy value in that group's runtime data.
The sidecar then starts at 32 and probes measured throughput inside this real
vLLM capacity. Later restarts reuse the recorded value, so they do not repeat
the discovery unless that value itself no longer boots.

For Docker groups, the sidecar is not considered healthy until that final
capacity marker exists *and* the proxy health endpoint responds. The marker is
cleared before every vLLM bootstrap, so a stale value cannot start the miner
against a transient 32/64/128 capacity-probe instance.

If a TP bootstrap fails, the launcher terminates the complete vLLM process
group and waits until the GPUs visible to that sidecar are effectively idle
before retrying or allowing Supervisor to restart it. This prevents a failed
rank from making the immediate next attempt falsely report insufficient VRAM.
The defaults are a 120-second cleanup wait and 512 MiB idle threshold; change
them only for hosts whose display stack has a known larger baseline.

No concurrency settings are required in `miner.env`. To view the decision:

```bash
set -a && source miner.env && set +a
CID=tensorcash-rig-01-g1-sidecar-1
docker exec -e NOMP_SIDECAR_TOKEN="$NOMP_SIDECAR_TOKEN" "$CID" sh -lc \
  'curl -fsS -H "Authorization: Bearer $NOMP_SIDECAR_TOKEN" http://127.0.0.1:8080/v1/tensorcash/metrics'
```

The `adaptive_concurrency` object reports the active level, configured range,
last probe/rollback decision, and local request-error count. The useful measure is
the sustained `generation_tokens_per_sec` plus accepted shares—not a
momentary GPU-power reading.

For a deliberate fixed-profile benchmark only, opt out explicitly:

```bash
TENSORCASH_CONCURRENCY_MODE=manual
VLLM_MAX_NUM_SEQS=32
NOMP_SIDECAR_CONCURRENCY=32
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=16
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=64
```

## Updating without re-downloading the model

```bash
bash start.sh --update
```

The update command pulls `mainnet-latest` on ordinary GPUs and
`mainnet-blackwell-latest` on Blackwell GPUs, records its immutable digest in
the host-local `miner.env`, and recreates the sidecar/miner containers. Docker
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
glibc 2.35, then mounts it over the runtime image's controller. This keeps
Ubuntu 22.04 and HiveOS Docker hosts compatible without downloading a second
large runtime image. Actual inference runs in the CUDA/vLLM sidecar, so GPU
compatibility is determined by that pinned runtime and available VRAM, not by
the Rust binary. The standard image supports the previously tested GPU tiers;
RTX 50-series/Blackwell requires the separate `mainnet-0.1.1-blackwell` image
because its PyTorch and vLLM CUDA extensions contain `sm_120` code. ARM64 and
untested GPU generations are not advertised as supported.
## Native mode (Vast/hosted containers without Docker)

Use this mode only when no Docker-compatible runtime is available. It needs a
root Ubuntu 22.04-style container, Python 3.10, around 35 GiB free disk, and
one or more clean GPUs with at least 22 GiB VRAM each. The first run builds the
public native runtime and downloads the pinned model; later starts reuse both.

```bash
git clone https://github.com/Avalonbtc/tensorcash-miner-launcher.git ~/tensorcash-miner
cd ~/tensorcash-miner
bash native-vast.sh \
  --pool pool.example.org:3336 \
  --wallet 'YOUR_PAYOUT_ADDRESS' \
  --worker 'vast-4090-01'
```

Native `auto` mode starts one independent TP=1 group for every eligible GPU.
For example, an 8x48 GiB rig starts `vast-4090-01-g1` through `-g8`, with
ports `8080` through `8087`. To restrict a host deliberately, add
`TENSORCASH_NATIVE_GPU_GROUPS=0,2,5` to `miner.env`. Each group has an
isolated sidecar, controller, proof data, PID files, and logs, while all groups
share the model cache and installed Python runtime. The first matching
GPU-model/VRAM group performs capacity discovery; its verified vLLM limit is
reused by later identical groups, which still fall back to local discovery if
the shared value cannot boot.

To update the native scheduler/controller overlays and restart without
downloading the model again:

```bash
cd ~/tensorcash-miner
git pull --ff-only
bash native-vast.sh --stop
bash native-vast.sh
```

Native mode copies the current launcher-owned sidecar into its installed
runtime on every normal start. This includes the concurrent-proof de-duplication
fix; no `--rebuild-runtime` or model re-download is needed for that update.

```bash
# Liveness, local sidecar health, and GPU utilisation/power.
bash native-vast.sh --status

# Follow controller, proxy, and vLLM logs together.
bash native-vast.sh --logs

# Stop native vLLM, sidecar, and controller.
bash native-vast.sh --stop

# Show group 1's authenticated sidecar scheduler and generation metric.
set -a && source runtime/native/instances/g1/runtime.env && set +a
curl -fsS -H "Authorization: Bearer $NOMP_SIDECAR_TOKEN" \
  http://127.0.0.1:8080/v1/tensorcash/metrics
```

See [NATIVE_VAST.md](NATIVE_VAST.md) for the native dependency and profile
details. Native mode is TP=1 only; use Docker mode for 8/12/16 GiB cards or
multi-GPU tensor parallelism within one vLLM process.
