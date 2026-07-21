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

For a 24 GiB 4090, begin with the default one-way inference profile. After it
has a sustained accepted-share baseline, edit `miner.env` and progress through
8, 16, 32, then at most 64. Keep the three settings consistent:

```bash
VLLM_MAX_NUM_SEQS=32
NOMP_SIDECAR_CONCURRENCY=32
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=16
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=128
TENSORCASH_SUBMIT_WINDOW=32
```

Native mode currently deliberately uses one TP=1 GPU per launcher directory.
For 8/12/16 GiB cards or multi-GPU tensor parallelism, use the regular Docker
launcher on a host that provides the NVIDIA Container Toolkit.
