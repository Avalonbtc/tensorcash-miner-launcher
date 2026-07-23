# TensorCash miner launcher

This is the public, source-free launcher for the TensorCash miner. It contains
only Docker Compose and shell scripts. The private build pipeline publishes a
stripped miner binary inside `ghcr.io/avalonbtc/tensorcash-miner`; no Rust
checkout, wallet, pool configuration, or model cache is published here.

## Choose a launch mode

- **Docker mode:** use `start.sh` on normal Linux/HiveOS hosts with Docker
  Compose v2 and NVIDIA Container Toolkit. It automatically starts 12--21 GiB
  cards as independent FP8 miners (using a serialized FP8 checkpoint at the
  12 GiB tier) and >=22 GiB cards as independent BF16 miners.
- **Native mode:** use `native-vast.sh` only in hosted GPU containers that
  expose `/dev/nvidia*` but intentionally provide no Docker daemon. It uses
  one independent TP=1 instance for every selected >=12 GiB GPU, selects
  serialized FP8 for 12--14.9 GiB cards, FP8 for 16--21 GiB cards, and BF16
  for >=22 GiB cards automatically.

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
It cannot execute on an RTX 5090 (`sm_120`). The separate
`ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.1-blackwell` image must be
published successfully before Docker mode can support a compute-capability
12.x host. Until its registry manifest exists, `start.sh` stops with a clear
message instead of writing an unusable image tag into `miner.env`; use native
mode for that host in the meantime. Custom tags and immutable digests are never
changed automatically.

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
| >=22 GiB | BF16 TP=1 per card | Every card becomes an independent miner group. |
| 15-21 GiB | FP8 TP=1 per card | Every card becomes an independent miner group. |
| 12-14.9 GiB | Serialized FP8 TP=1 per card | Downloads the pinned official static FP8 snapshot once, then every card becomes an independent group. |
| 6-11.4 GiB | FP8 TP=2 pairs | Uses 2, 4, 6, or 8 cards; an odd last card waits idle. |
| <6 GiB | Unsupported in auto mode | The mainnet 8B profile has insufficient FP8 headroom. |

For example, three 12 GB cards become `0;1;2`, four become `0;1;2;3`; and
eight 8 GB cards become `0,1;2,3;4,5;6,7`. Set
`TENSORCASH_MODEL_PRECISION=bf16` to deliberately retain the old TP grouping,
or `fp8` to require FP8 on every eligible group.

The ordinary Qwen3-8B checkpoint incurs a BF16-to-FP8 conversion peak while
vLLM constructs the model. That peak does not fit on a 12 GiB single card, so
lowering concurrency cannot fix it. For this tier the launcher instead
downloads the immutable public `Qwen/Qwen3-8B-FP8` snapshot at
`220b46e3b2180893580a4454f21f22d3ebb187d3`; its FP8 configuration and scales
are validated before use. Proof metadata remains the chain-pinned
`Qwen/Qwen3-8B@9c925d64...` profile—the local checkpoint path only controls
how vLLM loads the identical model family. The fallback remains TP=2 FP8 for
6/8 GiB pairs.

The first launch pulls the public runtime image and downloads the chain-pinned
Qwen3-8B snapshot once. Image pulls and model downloads retry automatically;
completed Docker layers and Hugging Face cache chunks are reused after an
interruption. It writes the model to
`runtime/models` by default. Every sidecar group on that physical host mounts
the same directory, so four GPUs or two groups still use one 16 GB disk cache.
Docker similarly stores the runtime image layers once per host.

### Automatic launcher repair and update

Every mining start force-syncs the launcher repository to `origin/main` before
starting containers. A copied package with missing or damaged `.git` metadata
is repaired from the public launcher repository first. This resets launcher
scripts and removes untracked launcher files, while preserving ignored
`miner.env`, `runtime/`, model caches, and logs. The updated script is then
re-executed in the same command.

Set `TENSORCASH_AUTO_UPDATE=false` only for an emergency offline recovery.
Status, logs, plan, and stop commands intentionally do not contact GitHub.

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
image/model transfer and verifies appended data. A 12 GiB FP8-only seed host
exports its verified `Qwen3-8B-FP8` snapshot instead of incorrectly requiring
the absent BF16 cache:

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

The seed script packages the loaded runtime image, the usable verified model
cache (canonical or serialized FP8), and the public launcher scripts. It
deliberately excludes `miner.env` and `runtime/data`, so no payout address,
sidecar token, or machine-local proof state is copied. It creates checksums and
binds the loaded image ID in the manifest, then optionally transfers the
complete bundle through resumable `rsync`.

On the fully downloaded seed host:

```bash
cd ~/tensorcash-miner
git pull --ff-only
bash seed-export.sh --copy-to root@DESTINATION_HOST:/root/
```

On the destination host, run the `seed-install.sh` path printed by the export
command. The directory suffix varies by image tag and model profile, for
example `mainnet-0.1.0-serialized-fp8` for a 12 GiB seed host:

```bash
bash /root/tensorcash-seed-mainnet-0.1.0-serialized-fp8/seed-install.sh
```

The installer verifies every file and image ID, installs `zstd` automatically
on Ubuntu if needed, loads the local Docker image, validates/extracts the local
model cache, prompts for the payout account, and starts the GPU groups. Its
first launch is offline: it skips only that first Git sync and registry pull;
later ordinary starts resume normal launcher updates. It uses the pool endpoint
from the seed host by default. Override values for non-interactive deployment:

```bash
bash /root/tensorcash-seed-mainnet-0.1.0-serialized-fp8/seed-install.sh \
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

A capacity that boots is not automatically safe for a full proof cohort. If a
running vLLM group exits, the launcher waits for its workers to release the
GPU, lowers only that group's effective sequence capacity by 64, records the
new healthy value, and restarts it after a short backoff. The sidecar reads the
same marker and immediately clamps its scheduler to that capacity. While the
local vLLM endpoint is unavailable, it pauses the affected job, uses a single
exponentially backed-off recovery probe, and never creates a retry storm.
`NOMP_SIDECAR_ADMISSION_SPREAD_MS` is automatic by default (12 ms per initial
request, capped at 30 seconds): initial work is dephased, but replacement work
is submitted immediately so the GPU remains occupied.

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
one clean GPU with at least 12 GiB VRAM or a valid TP pair. The first run builds
the public native runtime and downloads the pinned model; later starts reuse both.

```bash
git clone https://github.com/Avalonbtc/tensorcash-miner-launcher.git ~/tensorcash-miner
cd ~/tensorcash-miner
bash native-vast.sh \
  --pool pool.example.org:3336 \
  --wallet 'YOUR_PAYOUT_ADDRESS' \
  --worker 'vast-4090-01'
```

Native `auto` mode starts one independent TP=1 group for every eligible GPU.
It selects serialized FP8 for 12--14.9 GiB cards, FP8 for 16--21 GiB cards,
BF16 for >=22 GiB cards, and FP8 TP=2 groups for pairs of 6/8 GiB cards.
For example, an 8x48 GiB rig starts `vast-4090-01-g1` through `-g8`. It uses
`8080` upward for sidecars and automatically skips any local port triplet
already occupied by another host service. To restrict a host deliberately, add
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

In native adaptive mode, the vLLM batched-token budget is automatically
`8192` on 22--39 GiB cards and `65536` on cards with at least 40 GiB VRAM.
This is separate from the request-concurrency ceiling: it prevents high-VRAM
cards from being artificially held near one hundred active requests. High-VRAM
cards still begin at their configured adaptive start value; the controller
raises concurrency only after a full measured interval. For a deliberate
benchmark, set `TENSORCASH_AUTO_MAX_BATCHED_TOKENS` in `miner.env`; vLLM still
applies its own runtime memory-admission guard.

The high-VRAM profile also uses a 2048-proof local completion buffer so its
large in-flight cohort cannot stall behind short network submission bursts.
Its NOMP HTTP connection pool automatically matches the vLLM sequence ceiling
plus the prefetch reserve (rather than aiohttp's unrelated default of 100), so
the proxy can actually deliver the configured local concurrency. Advanced
benchmarking can override this with `NOMP_SIDECAR_HTTP_CONNECTIONS` (1--2048).
Native mode raises its child-process open-file limit to `65535` by default;
this is required because a 1024-request local profile uses more than the
common shell default of 1024 descriptors. Use `TENSORCASH_NATIVE_NOFILE_LIMIT`
only when the host has a stricter, verified policy.

The PoW proof-row pool is automatically sized to the vLLM running ceiling plus
its waiting reserve, so a full request batch cannot evict its own rows. This
affects local bookkeeping capacity only; it does not alter proof content or
consensus rules.

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
details. Native mode automatically downloads serialized FP8 for 12--14.9 GiB
cards and supports TP=2 FP8 pairs for 6/8 GiB cards; use Docker mode for
larger TP topologies.
