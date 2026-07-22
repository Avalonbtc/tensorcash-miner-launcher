# TensorCash native mode for Vast containers without Docker

Use this only when the rented GPU container exposes `/dev/nvidia*` but does
not provide `docker`, `podman`, `nerdctl`, `apptainer`, or another OCI runtime.
It starts the exact TensorCash proof path natively: the TensorCash vLLM overlay,
C++ proof processor, miner-proxy sidecar, and checksum-pinned controller.

The first install downloads Python/CUDA dependencies, public TensorCash source,
the 16 GiB chain-pinned model, and builds native extensions. It needs a root
Ubuntu 22.04-style container, Python 3.10, approximately 35 GiB free disk, and
one or more clean GPUs with at least 22 GiB VRAM each. It does not need Docker
and does not download any private Rust source.

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
least 22 GiB VRAM receives its own TP=1 group. An 8x48 GiB host therefore runs
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
it is not a fixed 24 GiB or 22 GiB tier. Every 60 seconds the sidecar keeps a
higher 16-request probe only when rolling generation throughput improves by at
least 2%, and rolls back on a 5% regression or local vLLM error. No concurrency
values need to be added to `miner.env`.

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
avoid a short submission burst starving the 1024 running requests.

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

Native mode uses one TP=1 process per eligible GPU. For 8/12/16 GiB cards or
multi-GPU tensor parallelism inside one vLLM process, use the regular Docker
launcher on a host that provides the NVIDIA Container Toolkit.
