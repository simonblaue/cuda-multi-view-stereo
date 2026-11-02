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
#  on the *current* CUDA device.  This host has a single GPU (GTX 1080 Ti,
#  sm_61), so no pinning is needed; `--gpus all` exposes it as device 0.
#
#  File ownership: images start as root, but the requested command is re-exec'd
#  as a non-root user so that anything written into the bind-mounted working
#  copy (the repo, its build/ and output/ dirs, ...) is owned by the host user
#  who invoked Docker, not by root.  Target uid/gid resolution priority:
#    1. CUMVS_UID / CUMVS_GID exported from .env (explicit override)
#    2. owner of /workspace/cumvs, the bind-mounted repo (== the invoker)
#    3. root (0:0), when running the prebuilt image without a mount
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
	log "rebuild with      : cmake -S /workspace/cumvs -B /workspace/cumvs/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=61 -DOpenCV_DIR=\${OpenCV_DIR} && cmake --build /workspace/cumvs/build -j"
fi

# -----------------------------------------------------------------------------
#  Ownership handling: resolve the target uid/gid and drop privileges so files
#  written to the bind mount belong to the invoking host user (or the override
#  in .env), never to root.
# -----------------------------------------------------------------------------
DROP_USER="${CUMVS_USER:-appuser}"
CUMVS_UID="${CUMVS_UID:-}"
CUMVS_GID="${CUMVS_GID:-}"

# Fallback to the owner of the bind-mounted working copy (the invoker).
if [ -z "$CUMVS_UID" ] || [ -z "$CUMVS_GID" ]; then
	if [ -d /workspace/cumvs ]; then
		read -r _auto_uid _auto_gid < <(stat -c '%u %g' /workspace/cumvs 2>/dev/null || printf '0 0')
		[ -z "$CUMVS_UID" ] && CUMVS_UID="$_auto_uid"
		[ -z "$CUMVS_GID" ] && CUMVS_GID="$_auto_gid"
	fi
fi
: "${CUMVS_UID:=0}"
: "${CUMVS_GID:=0}"

if [ "$CUMVS_UID" != "0" ] || [ "$CUMVS_GID" != "0" ]; then
	# Create the numeric primary group (idempotent).
	if ! getent group "$CUMVS_GID" >/dev/null 2>&1; then
		groupadd -g "$CUMVS_GID" "$DROP_USER" 2>/dev/null || true
	fi
	# Create a passwd entry (idempotent) so tools get a $HOME / username.
	if ! getent passwd "$CUMVS_UID" >/dev/null 2>&1; then
		useradd -u "$CUMVS_UID" -g "$CUMVS_GID" -o -m -d "/home/$DROP_USER" -s /bin/bash "$DROP_USER" 2>/dev/null || true
	fi
	mkdir -p "/home/$DROP_USER" 2>/dev/null || true
	chown -R "$CUMVS_UID:$CUMVS_GID" "/home/$DROP_USER" 2>/dev/null || true
	export HOME="/home/$DROP_USER"

	# Only here, while still root: if this host exposes the GPU device nodes
	# more restrictively than 0666, hand them to the target user so the CUDA
	# driver stays reachable after the drop.  (Most hosts already use 0666.)
	chown "$CUMVS_UID:$CUMVS_GID" /dev/nvidia* /dev/nvidia-caps/* 2>/dev/null || true

	log "running as uid=$CUMVS_UID gid=$CUMVS_GID ($DROP_USER, HOME=$HOME)"
	exec setpriv --reuid "$CUMVS_UID" --regid "$CUMVS_GID" --clear-groups "$@"
fi

exec "$@"
