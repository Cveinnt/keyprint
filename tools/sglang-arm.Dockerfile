# Derived from SGLang docker/arm64.Dockerfile at 13d593b6cf885c5c4d50eea88c82b9e28cf5941e.
# SPDX-License-Identifier: Apache-2.0
# Trusted offline ARM Docker pilot only; skips optional NUMA memory binding.
FROM ubuntu:24.04@sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9 AS runtime
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG VER_SGLANG=13d593b6cf885c5c4d50eea88c82b9e28cf5941e

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    gcc \
    g++ \
    make \
    cmake \
    libsqlite3-dev \
    google-perftools \
    libtbb-dev \
    libnuma-dev \
    numactl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    source $HOME/.local/bin/env && \
    uv venv --python 3.12

RUN echo -e '[[index]]\nname = "torch"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchvision"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchaudio"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "triton"\nurl = "https://download.pytorch.org/whl/cpu"' > .venv/uv.toml

ENV UV_CONFIG_FILE=/opt/.venv/uv.toml
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
# Avoid retaining a second copy of large wheels in image layers.
ENV UV_NO_CACHE=1

WORKDIR /sgl-workspace
RUN source $HOME/.local/bin/env && \
    source /opt/.venv/bin/activate && \
    git clone --filter=blob:none ${SGLANG_REPO} sglang && \
    cd sglang && \
    git checkout ${VER_SGLANG} && \
    cd python && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install . && \
    uv pip install 'scikit-build-core>=0.10' wheel

ENV SGLANG_USE_CPU_ENGINE=1
RUN echo 'source /opt/.venv/bin/activate' >> /root/.bashrc

WORKDIR /sgl-workspace/sglang

RUN /opt/.venv/bin/python - <<'PYFIX'
from pathlib import Path
p=Path('/sgl-workspace/sglang/python/sglang/kernels/aot/csrc/cpu/numa_utils.cpp')
s=p.read_text()
start=s.index('  // Memory node binding')
end=s.index('  // OMP threads binding', start)
p.write_text(s[:start] + '  // Keyprint ARM Docker pilot: NUMA topology is unavailable in this VM.\n  // Keep CPU thread binding; skip optional memory-node binding.\n\n' + s[end:])
PYFIX
# Build the patched kernel once against its already-installed, upstream-pinned
# torch dependency. Build isolation would unpack a second torch environment.
RUN source /root/.local/bin/env && source /opt/.venv/bin/activate && \
    cd /sgl-workspace/sglang/python/sglang/kernels/aot && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install --no-build-isolation .
FROM runtime AS pilot
ARG KEYPRINT_RUN_ID
COPY --from=keyprint_source /src /keyprint/src
COPY --from=keyprint_source /tools/validate_sglang.py /keyprint/validate_sglang.py
COPY --from=model_assets / /model
RUN --network=none --mount=type=secret,id=keyprint_key,required=true \
    test -n "${KEYPRINT_RUN_ID}" && \
    mkdir -m 700 /results && \
    printf '%s\n' "${KEYPRINT_RUN_ID}" > /results/run-id.txt && \
    PYTHONPATH=/keyprint/src KEYPRINT_MODEL_PATH=/model KEYPRINT_KEY_FILE=/run/secrets/keyprint_key KEYPRINT_TRACE_DIR=/results/traces OMP_NUM_THREADS=2 \
    timeout 180 /opt/.venv/bin/python /keyprint/validate_sglang.py > /results/pilot.log 2>&1; result=$?; echo $result > /results/exit-code.txt; cat /results/pilot.log
COPY --from=keyprint_source /tools/test_sglang_contract.py /keyprint/test_sglang_contract.py
RUN PYTHONPATH=/keyprint/src /opt/.venv/bin/python /keyprint/test_sglang_contract.py > /results/contract.log 2>&1; result=$?; echo $result > /results/contract-exit-code.txt; cat /results/contract.log
RUN /root/.local/bin/uv pip freeze --python /opt/.venv/bin/python > /results/runtime-lock.txt && sha256sum /sgl-workspace/sglang/python/sglang/kernels/aot/csrc/cpu/numa_utils.cpp > /results/numa-source-sha256.txt
FROM scratch AS receipts
COPY --from=pilot /results /

# Larger ordinary/marked comparison using the same pinned CPU runtime. Export
# results directly, avoiding another copy of the full image on the host.
FROM runtime AS comparisons
ARG KEYPRINT_RUN_ID
COPY --from=keyprint_source /src /keyprint/src
COPY --from=keyprint_source /tools /keyprint/tools
COPY --from=model_assets / /model
RUN --network=none --mount=type=secret,id=keyprint_key,required=true \
    test -n "${KEYPRINT_RUN_ID}" && \
    mkdir -m 700 /results && \
    printf '%s\n' "${KEYPRINT_RUN_ID}" > /results/run-id.txt && \
    /opt/.venv/bin/python -c "import os,secrets; f=os.open('/results/control.key',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(f,secrets.token_bytes(32)); os.close(f)" && \
    PYTHONPATH=/keyprint/src KEYPRINT_MODEL_PATH=/model KEYPRINT_KEY_FILE=/run/secrets/keyprint_key KEYPRINT_TRACE_DIR=/results/traces OMP_NUM_THREADS=2 \
    timeout --kill-after=15 900 /opt/.venv/bin/python /keyprint/tools/validate_native_cases.py --backend sglang > /results/comparison.log 2>&1; result=$?; echo $result > /results/exit-code.txt; cat /results/comparison.log
RUN PYTHONPATH=/keyprint/src /opt/.venv/bin/python /keyprint/tools/test_sglang_contract.py > /results/contract.log 2>&1; result=$?; echo $result > /results/contract-exit-code.txt; cat /results/contract.log
RUN /root/.local/bin/uv pip freeze --python /opt/.venv/bin/python > /results/runtime-lock.txt && \
    sha256sum /sgl-workspace/sglang/python/sglang/kernels/aot/csrc/cpu/numa_utils.cpp > /results/numa-source-sha256.txt
FROM scratch AS comparison-receipts
COPY --from=comparisons /results /
