# Dockerfile — Hailo Dataflow Compiler for best.onnx → best.hef
#
# Build & run (from the project root):
#   docker build -t hailo-compile hailo_compile/
#   docker run --rm \
#     -v "$(pwd)/best.onnx:/work/best.onnx:ro" \
#     -v "$(pwd):/out" \
#     hailo-compile
#
# The compiled best.hef will appear in the project root on the host.
#
# NOTE: You must supply hailo_dataflow_compiler-*.whl yourself.
#       Download it from https://hailo.ai/developer-zone/ (free account).
#       Place the .whl file in this hailo_compile/ directory before building.

FROM python:3.10-slim

WORKDIR /work

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Hailo DFC wheel — must be placed next to this Dockerfile
# If the file is not found the build will fail with a clear message.
COPY hailo_dataflow_compiler-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/hailo_dataflow_compiler-*.whl \
    && pip install --no-cache-dir opencv-python-headless numpy

# Copy compile script
COPY compile.py /work/compile.py

# Run compilation
# --onnx and --out use bind-mounted paths
CMD ["python", "compile.py", \
     "--onnx", "/work/best.onnx", \
     "--out",  "/out/best.hef", \
     "--har-dir", "/tmp/har"]
