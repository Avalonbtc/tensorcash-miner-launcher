# TensorCash native mode for Vast containers without Docker

Use this only when the rented GPU container exposes `/dev/nvidia*` but does
not provide `docker`, `podman`, `nerdctl`, `apptainer`, or another OCI runtime.
It starts the exact TensorCash proof path natively: the TensorCash vLLM overlay,
C++ proof processor, miner-proxy sidecar, and checksum-pinned controller.

The first install downloads Python/CUDA dependencies, public TensorCash source,
the 16 GiB chain-pinned model, and builds native extensions. It needs a root
Ubuntu 22.04-style container, Python 3.10, approximately 35 GiB free disk, and
one or more clean GPUs with at least 11.5 GiB VRAM each. It does not need Docker
and does not download any private Rust source.

## RTX 50-series / Blackwell

An RTX 50-series card is detected from compute capability `12.x`. It does not
try to run the old vLLM 0.10 Python wheel. Instead, the native launcher creates
an isolated `runtime/native/blackwell` environment, installs CUDA Toolkit 13
when it is not already present, installs the matching CUDA-13 PyTorch wheel,
and builds TensorCash' pinned vLLM 0.19 source for `sm_120` directly on that
host. The result is cached as a wheel and reused after restarts; it never
overwrites an existing 30/40-series native runtime.

The first Blackwell install is a real CUDA/C++ compile and can take hours.
Default parallelism is two build jobs to avoid exhausting ordinary rental
containers. On a host with ample CPU RAM, set `TENSORCASH_BLACKWELL_BUILD_JOBS`
to a verified value from `1` to `8`; do not increase it blindly. `--install`
performs the one-time preparation without beginning mining:

```bash
cd ~/tensorcash-miner
TENSORCASH_BLACKWELL_BUILD_JOBS=2 bash native-vast.sh --install
bash native-vast.sh
```

If CUDA Toolkit 13 is already installed, set `TENSORCASH_BLACKWELL_CUDA_HOME`
to its prefix when it is not `/usr/local/cuda-13.0` or `/usr/local/cuda`.
To disallow automatic package installation, set
`TENSORCASH_BLACKWELL_AUTO_INSTALL_CUDA_TOOLKIT=false`; the launcher then
fails before changing the system if a compatible `nvcc` is absent.

```bash
git clone https://github.com/Avalonbtc/tensorcash-miner-launcher.git ~/tensorcash-miner
cd ~/tensorcash-miner

bash native-vast.sh \
  --pool 37.59.104.113:3336 \
  --wallet 'YOUR_TENSORCASH_ADDRESS' \
  --worker 'vast-4090-01'
```

`miner.env` is created once with mode `0600`. Later starts reuse the built
runtime and model cache:

```bash
cd ~/tensorcash-miner
bash native-vast.sh
```

By default `TENSORCASH_NATIVE_GPU_GROUPS=auto`, so every visible card with at
least 11.5 GiB VRAM receives its own TP=1 group. 12/16 GiB cards use FP8 while
>=22 GiB cards use BF16. An 8x48 GiB host therefore runs
eight independent vLLM/proxy/controller pipelines. They share the source,
venv, controller binary and model weights, but have isolated ports, logs,
proof data and worker names (`<WORKER>-g1`, `<WORKER>-g2`, ...).

Use a comma-separated list to restrict the machine without editing launch
commands:

```bash
sed -i -E 's/^TENSORCASH_NATIVE_GPU_GROUPS=.*/TENSORCASH_NATIVE_GPU_GROUPS=0,2,5/' miner.env
bash native-vast.sh --stop
bash native-vast.sh
```

Group 1 prefers the familiar local ports `8000`, `8080`, and `7002`; each later
group prefers the next port in all three ranges. Before starting, the launcher
checks the full triplet and skips any occupied host ports, so unrelated local
services cannot make a later GPU fail to boot. On the first start of a hardware
profile, group 1 performs the conservative 32-to-capacity vLLM discovery. Its
verified capacity is shared by later groups with the same GPU model, VRAM,
model commit, memory-utilization setting and requested ceiling. If a card
cannot boot the shared value, its local vLLM bootstrap automatically discards
it and performs the normal safe fallback discovery.

Useful operations:

```bash
bash native-vast.sh --status
bash native-vast.sh --logs
bash native-vast.sh --stop
bash native-vast.sh --install
bash native-vast.sh --rebuild-runtime
```

To apply a launcher-side scheduler update without rebuilding Python extensions
or downloading the model again, stop, pull, and start normally. Each normal
native start re-installs the public NOMP sidecar overlay into
`runtime/native/miner-proxy/src`, including concurrent-proof de-duplication.

```bash
cd ~/tensorcash-miner
git pull --ff-only
bash native-vast.sh --stop
bash native-vast.sh
```

Native mode now starts at 32 requests automatically and discovers the highest
bootable vLLM capacity in ascending steps, up to the 1024 engineering safety
ceiling. The final value depends on the model, GPU, driver and available VRAM;
it is not a fixed 24 GiB or 22 GiB tier. A bootable value is verified again
under real proof load: if vLLM later exits, that group waits for VRAM cleanup,
falls back by 64 sequences, saves the new effective value, and restarts only
that group. The sidecar consumes the same saved value and pauses with one
backed-off recovery probe rather than spinning thousands of failed requests.
No concurrency values need to be added to `miner.env`.

The related vLLM batched-token scheduler budget is automatic too: 22--39 GiB
cards retain the validated `8192` value, while >=40 GiB cards receive `65536`.
Without this VRAM-aware budget, `VLLM_MAX_NUM_SEQS=1024` can still admit only
about one hundred active requests on a 48 GiB card. Advanced benchmark users
can override it with `TENSORCASH_AUTO_MAX_BATCHED_TOKENS`; vLLM remains the
final memory-safety authority.

On >=40 GiB profiles, native auto mode also begins at its configured
concurrency ceiling rather than slowly re-probing from 32 after a restart. A
local vLLM request error or sustained regression still triggers the existing
adaptive rollback. That profile has a 2048-proof local completion buffer to
avoid a short submission burst starving its large running cohort. Initial
requests are automatically spread over a short admission window, while
replacement work is submitted immediately to prevent synchronized cohorts and
avoidable GPU-idle gaps.

The local NOMP HTTP connection pool is derived from the vLLM sequence ceiling
plus prefetch reserve. This avoids aiohttp's default 100-connection cap
silently limiting a high-concurrency profile. Set
`NOMP_SIDECAR_HTTP_CONNECTIONS` only for a controlled benchmark; accepted
shares and sustained generation rate remain the performance metric.

Native mode raises its soft `nofile` limit to `65535` before it starts any
vLLM, sidecar, or controller child. This is required for a 1024-request local
profile: the usual 1024-descriptor shell limit causes `socket.accept()` to
fail. Set `TENSORCASH_NATIVE_NOFILE_LIMIT` only for a host with a verified
alternative limit; a hard limit below 4096 is rejected before mining starts.

The proof sampler's row pool follows each group's vLLM sequence ceiling plus
its bounded wait reserve, preventing a full high-concurrency batch from
evicting rows from its own proof bookkeeping.

Use `bash native-vast.sh --status` for process/GPU state and this command for
the adaptive decision and rolling generation rate:

```bash
set -a && source runtime/native/instances/g1/runtime.env && set +a
curl -fsS -H "Authorization: Bearer $NOMP_SIDECAR_TOKEN" \
  http://127.0.0.1:8080/v1/tensorcash/metrics
```

For a deliberate fixed benchmark, set
`TENSORCASH_CONCURRENCY_MODE=manual` and then provide the matching
`VLLM_MAX_NUM_SEQS`, `NOMP_SIDECAR_CONCURRENCY`, and proof-buffer values.

## Automatic precision profile

The launcher resolves the profile for every selected GPU from its detected
VRAM: 12/16 GiB TP=1 cards use FP8 and >=22 GiB TP=1 cards use BF16. Both
launchers use the same `runtime-profile.sh` policy, and each instance writes
its resolved profile into `runtime.env` and its startup log.

The default is:

```bash
TENSORCASH_MODEL_PRECISION=auto
```

Set `TENSORCASH_MODEL_PRECISION=fp8` or `bf16` only for a deliberate forced
profile. The older `TENSORCASH_VLLM_QUANTIZATION=fp8` remains a compatible
FP8 override, but other quantizers are rejected by the launcher.
