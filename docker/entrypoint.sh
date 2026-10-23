#!/usr/bin/env bash
# =============================================================================
#  Entry point for the cuda-multi-view-stereo container.
#
#  Prints a short environment report (GPU count / CUDA runtime / OpenCV) and
#  then execs whatever was requested, so the container stays usable as a plain
#  interactive shell:
#
#      docker run --gpus all -it --rm -v "$PWD":/workspace/cumvs cumvs
#
#  Layout inside the image:
#    /opt/opencv        OpenCV 4.10.0 (cudev + viz)
#    /opt/cumvs-src     pristine source shipped with the image
#    /opt/cumvs-build   prebuilt binaries (on PATH, never shadowed by mounts)
#    /workspace/cumvs   optional bind mount of your working copy
#
#  Because the CUMVS library never calls cudaSetDevice(), every instance runs
#  on the *current* CUDA device.  To spread work over the three L40s, launch
#  one container per GPU and pin it with CUDA_VISIBLE_DEVICES:
#
#      docker run --gpus '"device=0"' -e CUDA_VISIBLE_DEVICES=0 ... cumvs
#      docker run --gpus '"device=1"' -e CUDA_VISIBLE_DEVICES=1 ... cumvs
#      docker run --gpus '"device=2"' -e CUDA_VISIBLE_DEVICES=2 ... cumvs
# =============================================================================
set -euo pipefail

log() { printf '[cumvs] %s\n' "$*"; }

if command -v nvidia-smi >/dev/null 2>&1; then
	if nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null; then
		:
	else
		log "nvidia-smi could not query the driver (NVML/driver mismatch is common"
		log "in containers); CUDA runtime calls still work."
	fi
else
	log "nvidia-smi not available in this image."
fi

if command -v nvcc >/dev/null 2>&1; then
	log "nvcc: $(nvcc --version | tail -n 1)"
fi

if [ -d /opt/opencv/lib/cmake/opencv4 ]; then
	version_file=/opt/opencv/lib/cmake/opencv4/OpenCVConfig-version.cmake
	version=$(sed -n 's/^set(OpenCV_VERSION *\(.*\))/\1/p' "$version_file" 2>/dev/null | head -n 1)
	log "OpenCV: ${version:-unknown} (${OpenCV_DIR:-/opt/opencv/lib/cmake/opencv4})"
fi

log "workdir: $(pwd)"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [ -x /opt/cumvs-build/samples/app_patch_match_mvs ]; then
	log "prebuilt binaries : /opt/cumvs-build/samples (also on PATH)"
	log "image source      : /opt/cumvs-src"
fi

# A bind-mounted working copy shadows nothing (the build tree lives in
# /opt/cumvs-build), but its binaries have to be rebuilt explicitly.
if [ -d /workspace/cumvs ] && [ -f /workspace/cumvs/CMakeLists.txt ]; then
	log "mounted source    : /workspace/cumvs"
	log "rebuild with      : cmake -S /workspace/cumvs -B /workspace/cumvs/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DOpenCV_DIR=\${OpenCV_DIR} && cmake --build /workspace/cumvs/build -j"
fi

exec "$@"
