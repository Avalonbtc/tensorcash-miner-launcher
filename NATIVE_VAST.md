# TensorCash native mode for Vast containers without Docker

Use this only when the rented GPU container exposes `/dev/nvidia*` but does
not provide `docker`, `podman`, `nerdctl`, `apptainer`, or another OCI runtime.
It starts the exact TensorCash proof path natively: the TensorCash vLLM overlay,
C++ proof processor, miner-proxy sidecar, and checksum-pinned controller.

The first install downloads Python/CUDA dependencies, public TensorCash source,
the 16 GiB chain-pinned model, and builds native extensions. It needs a root
Ubuntu 22.04-style container, Python 3.10, approximately 35 GiB free disk, and
one clean GPU with at least 22 GiB VRAM. It does not need Docker and does not
download any private Rust source.

```bash
git clone https://github.com/Avalonbtc/tensorcash-miner-launcher.git ~/tensorcash-miner
cd ~/tensorcash-miner

bash native-vast.sh \
  --pool 37.59.104.113:3336 \
  --wallet 'YOUR_TENSORCASH_ADDRESS' \
  --worker 'vast-4090-01' \
  --gpu 0
```

`miner.env` is created once with mode `0600`. Later starts reuse the built
runtime and model cache:

```bash
cd ~/tensorcash-miner
bash native-vast.sh
```

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

Native mode now starts at 32 requests automatically. A 24 GiB TP=1 GPU can
probe up to 128; a 22-23 GiB TP=1 GPU is capped at 64. Every 60 seconds the
sidecar keeps a higher 16-request probe only when rolling generation throughput
improves by at least 2%, and rolls back on a 5% regression or local vLLM error.
No concurrency values need to be added to `miner.env`.

Use `bash native-vast.sh --status` for process/GPU state and this command for
the adaptive decision and rolling generation rate:

```bash
set -a && source miner.env && set +a
curl -fsS -H "Authorization: Bearer $NOMP_SIDECAR_TOKEN" \
  http://127.0.0.1:8080/v1/tensorcash/metrics
```

For a deliberate fixed benchmark, set
`TENSORCASH_CONCURRENCY_MODE=manual` and then provide the matching
`VLLM_MAX_NUM_SEQS`, `NOMP_SIDECAR_CONCURRENCY`, and proof-buffer values.

Native mode currently deliberately uses one TP=1 GPU per launcher directory.
For 8/12/16 GiB cards or multi-GPU tensor parallelism, use the regular Docker
launcher on a host that provides the NVIDIA Container Toolkit.
