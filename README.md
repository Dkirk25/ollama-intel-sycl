# ollama-intel-sycl

Docker image for running [Ollama](https://github.com/ollama/ollama) on Intel Arc GPUs through the SYCL / oneAPI backend.

The image takes the official Ollama Linux release and adds a `ggml-sycl` backend built from the matching Ollama source, along with the current Intel GPU runtime. It also includes a fix for model reloads that hang on Intel `xe` GPUs.

Tested on an **Intel Arc Pro B70 (Battlemage G31)** under **Unraid** (kernel 6.18, `xe` driver).

## Versions

| Component | Version | Build arg |
|---|---|---|
| Ollama | 0.40.16 | `OLLAMA_VERSION` |
| Intel oneAPI DPC++ compiler + MKL | 2026.2 | `ONEAPI_VERSION` |
| oneDNN | 2026.1 | `DNNL_VERSION` |
| Intel compute-runtime | 26.40.40196.8 | `COMPUTE_RUNTIME_VERSION` |
| Intel Graphics Compiler (IGC) | 2.42.6 (build 23070) | `IGC_VERSION`, `IGC_BUILD` |
| gmmlib | 22.10.2 | `GMM_VERSION` |
| Level Zero loader | 1.35.0 | `LEVEL_ZERO_VERSION` |
| Base image | Ubuntu 24.04 | |

## Performance

These numbers come from an Arc Pro B70 with flash attention on, q8_0 KV cache, 8k context, a ~1.5k-token prompt and 256 generated tokens. Each figure is the median of 3 runs.

| Model | Prompt tok/s | Generation tok/s |
|---|---|---|
| llama3.1 8B | ~4,000 | ~88 |
| qwen3 14B | ~2,300 | ~50 |
| gpt-oss 20B | ~2,850 | ~92 |

The same models on the previous build (oneAPI 2025.2) ran at 1,300 / 1,093 / 2,350 prompt tok/s. Generation speed was about the same.

## Requirements

- An Intel Arc GPU (Alchemist or Battlemage) using the `xe` or `i915` kernel driver.
- The GPU's render node passed into the container, for example `/dev/dri/renderD128`.
- Your host's `render` group ID; it's `18` on Unraid. Check yours with `getent group render`.

## Build

```bash
docker build -t ollama-intel-sycl:0.40.16 .
```

The build compiles the SYCL backend with Intel's `icpx`, and each compile job uses about 2 GB of RAM. `BUILD_JOBS` defaults to 8 so the build fits in a 16 GB Docker VM. Raise it if you have more memory:

```bash
docker build --build-arg BUILD_JOBS=16 -t ollama-intel-sycl:0.40.16 .
```

To move a built image to another host without a registry:

```bash
docker save ollama-intel-sycl:0.40.16 | ssh user@host docker load
```

## Run

```bash
docker run -d --name ollama-intel-sycl \
  --device=/dev/dri/renderD128 --group-add 18 \
  -p 11434:11434 \
  -v /path/to/ollama:/root/.ollama \
  -v /path/to/ollama/cache:/root/.cache \
  -e OLLAMA_FLASH_ATTENTION=true \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  ollama-intel-sycl:0.40.16
```

Mount `/root/.cache` to a persistent folder. The Intel driver saves compiled GPU kernels there. With the folder persisted, only the **first** load of each model takes about 60 s while kernels compile, and later loads take a few seconds.

### Unraid

1. Add a container with **Repository** set to your image tag.
2. Set **Extra Parameters** to `--device=/dev/dri/renderD128 --group-add 18`.
3. Add path mappings for `/root/.ollama` (models) and `/root/.cache` (kernel cache).
4. Add the environment variables from the table below.

### Multiple GPUs

`ONEAPI_DEVICE_SELECTOR=level_zero:0` selects the first Level Zero device. If the host has several Intel GPUs, pass in only the render node you want, or change the index.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0:11434` | |
| `OLLAMA_NUM_GPU` | `999` | Offloads all layers to the GPU. |
| `OLLAMA_KEEP_ALIVE` | `10m` | `-1` keeps models loaded indefinitely. |
| `OLLAMA_NUM_PARALLEL` | `1` | |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0` | Selects which GPU to use. |
| `ZES_ENABLE_SYSMAN` | `1` | Lets Ollama read free VRAM. |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS` | `1` | |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | `1` | |

> ⚠️ **Do not set `SYCL_CACHE_PERSISTENT=1`.** With oneAPI 2025.3 or newer, it makes the first GPU kernel launch segfault (`ggml_sycl_op_set_rows` → `libsycl` `strcmp`), so every model load fails. Many older guides and templates set it, so remove it if you're upgrading. The driver's own cache in `/root/.cache` still keeps reloads fast.

## How it works

1. **Builder stage.** On Ubuntu 24.04, it installs oneAPI from Intel's apt repository, because Docker Hub's `intel/oneapi-basekit` images stop at 2025.3. It then clones Ollama at `v${OLLAMA_VERSION}` and builds only the `ggml-sycl` target from `llama/server`, using `icx`/`icpx`.
2. **Runtime bundling.** It copies both Level Zero adapters, `libur_adapter_level_zero` (V1) and `_v2`. On Battlemage, the oneAPI runtime picks V2 by default and loads it at runtime with `dlopen`, so `ldd` can't see it. Every other oneAPI library the backend links to is collected with `ldd`. The build **fails** if any dependency is missing from the bundle.
3. **Runtime stage.** It installs the Intel GPU packages (compute-runtime, IGC, gmmlib, Level Zero), unpacks the official Ollama release, and drops the SYCL backend into `/usr/lib/ollama/sycl/`. Ollama finds the backend there by itself.

## Reload hang fix

**Symptom:** when a model is unloaded and another load starts right after, about 1 in 4 loads hangs. Ollama gives up after 5 minutes with `timed out waiting for llama-server to start`, and the next attempt works.

**Cause:** Ollama stops its `llama-server` process with `SIGKILL`. If that process had just been running on the GPU, the Intel `xe` kernel driver doesn't always clean up properly. The next process's weight upload then never completes:

- Its main thread spins in `libze_intel_gpu` waiting for a GPU fence.
- The host kernel log shows `xe ... Engine reset: engine_class=bcs/ccs`.

**Reproduction** (`llama-server` on its own, with no Ollama):

| Test | Hangs |
|---|---|
| Load, then `SIGKILL` | 0 / 50 |
| Generate, then `SIGKILL` | 5 / 20 |
| Generate, then `SIGTERM` | 0 / 20 |

**Fix:** [`llama-server-wrapper.sh`](llama-server-wrapper.sh) is installed as `/usr/lib/ollama/llama-server`, and the real binary becomes `llama-server.real`. The wrapper starts the real server with `setpriv --pdeathsig TERM`. When Ollama kills the wrapper, the kernel sends the real server a graceful `SIGTERM`. `tini` runs as PID 1 to reap those orphaned processes.

**Result through Ollama:** 5 / 20 hangs before the fix and 0 / 30 after, with no further engine resets.

## Upgrading

1. Bump the `ARG` versions at the top of the `Dockerfile`.
   - **Intel GPU stack:** asset names follow the [compute-runtime release notes](https://github.com/intel/compute-runtime/releases). Each compute-runtime release lists the IGC version and build number it pairs with.
   - **oneAPI:** to see which versions exist, run `apt-cache search intel-oneapi-compiler-dpcpp-cpp` against `https://apt.repos.intel.com/oneapi`.
2. Build under a test tag.
3. **Test on the GPU before deploying.** At minimum:
   - The startup log shows `inference compute ... library="Intel(R) Arc(TM) ..."`.
   - A model loads with `offloaded N/N layers to GPU`.
   - A generation succeeds.
   - Several unload→reload cycles complete without hanging.
4. Deploy.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `total_vram="0 B"`, running on CPU | Render node not passed in, wrong group, or the `level_zero_v2` adapter missing (`UR_RESULT_ERROR_UNSUPPORTED_VERSION`). |
| Segfault on every model load | `SYCL_CACHE_PERSISTENT=1` is set. |
| First load takes ~60 s | Normal on the first run: kernels are compiling. Persist `/root/.cache` so it only happens once. |
| Build killed with `terminated by kill signal` | Out of memory. Lower `BUILD_JOBS`. |
| Load hangs ~5 min, then works | You're on an image without the wrapper fix. Check that `/usr/lib/ollama/llama-server` is the wrapper script. |

## Credits

- [Ollama](https://github.com/ollama/ollama) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- Intel [compute-runtime](https://github.com/intel/compute-runtime), [IGC](https://github.com/intel/intel-graphics-compiler), [Level Zero](https://github.com/oneapi-src/level-zero) and the [oneAPI toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html), each under its own license.
- Inspired by SpaceinvaderOne's [ollama-intel-gpu](https://github.com/SpaceinvaderOne/ollama-intel-gpu) Unraid template.

This is an unofficial community build and isn't affiliated with Ollama or Intel.
