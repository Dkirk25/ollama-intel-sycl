ARG OLLAMA_VERSION=0.40.16
ARG COMPUTE_RUNTIME_VERSION=26.40.40196.8
ARG LEVEL_ZERO_VERSION=1.35.0
ARG IGC_VERSION=2.42.6
ARG IGC_BUILD=23070
ARG GMM_VERSION=22.10.2
ARG ONEAPI_VERSION=2026.2
ARG DNNL_VERSION=2026.1

# =============================================================================
# Stage 1: Build ggml-sycl backend from Ollama source using Intel oneAPI
# oneAPI comes from Intel's apt repo: Docker Hub images stop at 2025.3.x.
# =============================================================================
FROM ubuntu:24.04 AS sycl-builder
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG OLLAMA_VERSION
ARG ONEAPI_VERSION
ARG DNNL_VERSION
# icpx SYCL compiles need ~2 GB each; nproc jobs OOM a 16 GB Docker VM
ARG BUILD_JOBS=8

RUN apt-get update && \
    apt-get install --no-install-recommends -q -y \
        ca-certificates curl gpg git cmake build-essential && \
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
        gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install --no-install-recommends -q -y \
        intel-oneapi-compiler-dpcpp-cpp-${ONEAPI_VERSION} \
        intel-oneapi-mkl-sycl-devel-${ONEAPI_VERSION} \
        intel-oneapi-dnnl-devel-${DNNL_VERSION} && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v${OLLAMA_VERSION} \
    https://github.com/ollama/ollama.git /ollama

WORKDIR /ollama

RUN source /opt/intel/oneapi/setvars.sh >/dev/null && \
    cmake -S llama/server -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_TARGET=INTEL \
    -DGGML_SYCL_F16=ON && \
    cmake --build build --parallel ${BUILD_JOBS} --target ggml-sycl && \
    mkdir -p /sycl-runner && \
    find /ollama/build -name "libggml-sycl.so" -exec cp {} /sycl-runner/ \;

# Bundle the oneAPI runtime: both L0 adapters (V2 is the default on Battlemage,
# and it is dlopen'ed so ldd can't see it), plus every oneAPI lib they link to.
RUN source /opt/intel/oneapi/setvars.sh >/dev/null && \
    L=/opt/intel/oneapi/compiler/latest/lib && \
    cp -P $L/libur_adapter_level_zero.so* $L/libur_adapter_level_zero_v2.so* /sycl-runner/ && \
    for f in /sycl-runner/libggml-sycl.so /sycl-runner/libur_adapter_level_zero*.so.0; do \
        ldd "$f" | awk '$3 ~ /^\/opt\/intel\// {print $3}'; \
    done | sort -u | xargs -I{} cp -L {} /sycl-runner/ && \
    (cp $L/libsycl-*.spv /sycl-runner/ 2>/dev/null || true) && \
    MISSING=$(LD_LIBRARY_PATH=/sycl-runner ldd /sycl-runner/libggml-sycl.so | grep "not found" | grep -v libggml || true) && \
    if [ -n "$MISSING" ]; then echo "$MISSING"; exit 1; fi && \
    (strip --strip-unneeded /sycl-runner/*.so* 2>/dev/null || true) && \
    ls -l /sycl-runner

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

ARG OLLAMA_VERSION
ARG COMPUTE_RUNTIME_VERSION
ARG LEVEL_ZERO_VERSION
ARG IGC_VERSION
ARG IGC_BUILD
ARG GMM_VERSION

RUN apt-get update && \
    apt-get install --no-install-recommends -q -y \
        ca-certificates \
        wget \
        zstd \
        ocl-icd-libopencl1 \
        libhwloc15 \
        tini && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/gpu && cd /tmp/gpu && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/libze1_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMM_VERSION}_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    dpkg -i *.deb && \
    rm -rf /tmp/gpu

RUN wget -qO- "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" | \
    zstd -d | tar -xf - -C /usr && \
    rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/vulkan && \
    mv /usr/lib/ollama/llama-server /usr/lib/ollama/llama-server.real

COPY --from=sycl-builder /sycl-runner/ /usr/lib/ollama/sycl/
# Turns Ollama's SIGKILL of llama-server into a graceful SIGTERM (see script)
COPY llama-server-wrapper.sh /usr/lib/ollama/llama-server
RUN sed -i 's/\r$//' /usr/lib/ollama/llama-server && chmod 755 /usr/lib/ollama/llama-server

ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_NUM_PARALLEL=1
ENV OLLAMA_KEEP_ALIVE=10m
ENV OLLAMA_NUM_GPU=999
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0
ENV ZES_ENABLE_SYSMAN=1
# Do NOT set SYCL_CACHE_PERSISTENT=1: with oneAPI >= 2025.3 it segfaults the
# first kernel launch (set_rows -> libsycl strcmp). The GPU driver's own
# kernel cache in /root/.cache still keeps reloads fast.
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
ENV UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
ENV no_proxy=localhost,127.0.0.1

EXPOSE 11434

VOLUME ["/root/.ollama"]

# tini reaps llama-server children orphaned by the wrapper after Ollama kills it
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/bin/ollama"]
CMD ["serve"]
