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

# Follow only the controller. It reports its explicit rolling window,
# rolling `generation=…tok/s` GPU performance rating, raw accepted shares/s,
# model-difficulty-normalized PoI/s, a separate
# network-target-equivalent estimate, sidecar queue pressure, and phase
# timing for sidecar start/claim/ack, proof delivery, and pool response.
# The report is every 30 seconds by default.
docker logs -f tensorcash-rig-01-g1-miner-1

# Stop all groups created by this launcher.
bash start.sh --stop
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
TENSORCASH_SUBMIT_WINDOW=4
```

After a stable real-pool run, a 24 GiB single-GPU profile can test 16, 32,
64, then 96 and 128 slots. These are throughput experiments, not a model or
consensus change:

```bash
# 16 slots
VLLM_MAX_NUM_SEQS=16
NOMP_SIDECAR_CONCURRENCY=16
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=8
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=32
TENSORCASH_SUBMIT_WINDOW=16

# 32 slots (only after the 16-slot profile remains zero-reject)
VLLM_MAX_NUM_SEQS=32
NOMP_SIDECAR_CONCURRENCY=32
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=16
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=64
TENSORCASH_SUBMIT_WINDOW=32

# 64 slots (only after the 32-slot profile remains zero-reject)
VLLM_MAX_NUM_SEQS=64
NOMP_SIDECAR_CONCURRENCY=64
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=32
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=128
TENSORCASH_SUBMIT_WINDOW=64

# 96 slots (24 GiB TP=1 only; benchmark for at least 10 minutes)
GPU_MEM_UTIL=0.90
VLLM_MAX_NUM_SEQS=96
NOMP_SIDECAR_CONCURRENCY=96
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=48
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=192
TENSORCASH_SUBMIT_WINDOW=64

# 128 slots (only retain it if the 5-minute generation rate improves)
GPU_MEM_UTIL=0.90
VLLM_MAX_NUM_SEQS=128
NOMP_SIDECAR_CONCURRENCY=128
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=64
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=256
TENSORCASH_SUBMIT_WINDOW=64
```

Keep `NOMP_SIDECAR_CONCURRENCY` less than or equal to
`VLLM_MAX_NUM_SEQS`, and keep `TENSORCASH_SUBMIT_WINDOW` at 64 or below. A
larger sidecar queue keeps the GPU full; the bounded 64-request submit window
prevents a slow pool confirmation path from multiplying network retries. The
scheduler leases proof ids before parallel pool submission and batch-acknowledges
only terminal results, so a reconnect cannot silently discard revenue and an
acknowledgement round does not drain the GPU queue to the old low-water mark.
Only use 96/128 on a single >=24 GiB GPU with `GPU_MEM_UTIL=0.90`; 12 GB TP=2
groups stay at the 64-or-lower profiles. Test each profile for at least ten
minutes, compare the rolling `generation=` rate, and keep the higher setting
only when it improves that rate without vLLM 5xx responses or rising rejects.
The scheduler automatically spreads each 96/128-slot request cohort over a
short sub-second interval. This avoids the saw-tooth pattern where all fixed
256-token requests complete together and briefly leave the GPU idle. It does
not alter a proof, model, target, or consensus rule. Leave this automatic
value enabled; only set `NOMP_SIDECAR_ADMISSION_SPREAD_MS=0` for an A/B test.
`NOMP_SIDECAR_PREFETCH_REQUESTS` is an optional experiment, disabled by
default. A reserve can let vLLM begin the next requests before a fixed-length
cohort returns through the sidecar, but some vLLM builds lose generation
throughput when a queue is present. Try `16` on a 128-slot 24 GiB profile
only after recording a stable zero-prefetch baseline; retain it solely when
the rolling generation rate improves without vLLM errors.
The scheduler deliberately rejects values above 128: hundreds of in-flight
256-token requests increase stale proof and verifier pressure without a linear
gain. The useful metric is sustained generation throughput plus accepted work,
not a momentary GPU power draw.

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
glibc 2.35, then mounts it over the runtime image's controller. This keeps
Ubuntu 22.04 and HiveOS Docker hosts compatible without downloading a second
large runtime image. Actual inference runs in the CUDA/vLLM sidecar, so GPU
compatibility is determined by that pinned runtime and available VRAM, not by
the Rust binary. RTX 4070 Super has been tested; ARM64 and untested GPU
generations are not advertised as supported.
# TensorCash miner launcher

Two launch modes are available:

- `bash start.sh ...` for normal GPU hosts with Docker + NVIDIA Container Toolkit.
- `bash native-vast.sh ...` for hosted containers which expose NVIDIA devices but
  deliberately do not provide a Docker daemon. See [NATIVE_VAST.md](NATIVE_VAST.md).
