# syntax=docker/dockerfile:1

# =============================================================================
#  cuda-multi-view-stereo (CUMVS) -- self-contained GPU development container
#
#  Host target : 3x NVIDIA L40 (Ada Lovelace, compute capability 8.9 / sm_89)
#
#  Baked-in dependencies
#    * CUDA Toolkit 12.4.1 (nvcc)              -> /usr/local/cuda
#    * OpenCV 4.10.0 + opencv_contrib 4.10.0  -> /opt/opencv
#        - built WITH_CUDA=ON  => `cudev` module + CUDA-enabled `core`
#          (`opencv2/core/cuda.hpp` / `cv::cuda::GpuMat` live in `core`)
#        - `viz` module (VTK-backed) is mandatory: samples/draw.h includes
#          <opencv2/viz.hpp>, and samples/CMakeLists.txt does
#          `find_package(OpenCV REQUIRED)` which resolves to *every* module
#          present in the installed OpenCVConfig.
#    * VTK 9.1 (Ubuntu package)                -> required by opencv_viz
#    * CMake / Ninja / g++ / OpenMP           -> build toolchain
#
#  Version compatibility notes (all verified, see AGENTS.md)
#    - OpenCV 4.10.0 is the first release containing the fix for opencv_contrib
#      issue #3690 ("CUDA Toolkit 12.4.0 `tuple` incompatibility"): cudev's
#      block/reduce.hpp guards the CCCL tuple change with
#      `#if (CUDART_VERSION < 12040)` and cudev/util/detail/tuple.hpp uses
#      `thrust::tuple`.  OpenCV 4.9.x does NOT build against CUDA 12.4.
#    - `cudev` and `viz` live in opencv_contrib, NOT in the main opencv repo.
#    - CUDA_ARCH_BIN=89 emits native SASS for the L40.  The repo default
#      "52;61;72;75;86" contains no sm_89 and would fall back to PTX JIT
#      (slow first launch, and fails outright on future drivers).
#    - The repo itself is single-GPU: it never calls cudaSetDevice(), so it
#      always runs on the *current* device.  Use CUDA_VISIBLE_DEVICES to pin
#      an instance to one of the three L40s and run three in parallel.
#
#  Build
#    docker build -t cumvs .
#
#  Run (all 3 GPUs visible, repo bind-mounted)
#    docker run --gpus all -it --rm -v "$PWD":/workspace/cumvs cumvs
#
#  Run (pin to GPU 1)
#    docker run --gpus '"device=1"' -it --rm -v "$PWD":/workspace/cumvs cumvs
#
#  Layout inside the image
#    /opt/opencv        OpenCV 4.10.0 (cudev + viz)
#    /opt/cumvs-src     pristine source shipped with the image
#    /opt/cumvs-build   prebuilt binaries, also on PATH
#    /workspace/cumvs   optional bind mount of your working copy
# =============================================================================

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG OPENCV_VERSION=4.10.0
# NVIDIA L40 = Ada Lovelace = compute capability 8.9.
# Other values: 80 (A100), 86 (RTX 30xx), 90 (H100).
ARG CUDA_ARCH_BIN=89
ARG CUDA_ARCH_PTX=89

# -----------------------------------------------------------------------------
# Stage 1: build OpenCV (+ contrib) with CUDA support and the VTK-backed viz
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS opencv-builder

ARG OPENCV_VERSION
ARG CUDA_ARCH_BIN
ARG CUDA_ARCH_PTX

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        pkg-config \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libopenjp2-7-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libgtk-3-dev \
        libvtk9-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# `cudev` and `viz` are only available in opencv_contrib, so both trees are
# required.  The sources are fetched into a cache mount so that rebuilding the
# image after a version bump does not re-download them.
RUN --mount=type=cache,target=/var/cache/git \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git /opt/opencv-src \
 && git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git /opt/opencv-contrib

# BUILD_LIST is the transitive closure of what this repo actually uses:
#   core      -> cuda_multi_view_stereo.h, core/quaternion.hpp, FileStorage
#   imgproc   -> resize / colour conversion
#   imgcodecs -> imread / imwrite (pulled in via highgui.hpp)
#   highgui   -> imshow / waitKey
#   viz       -> samples/draw.h  <opencv2/viz.hpp>
#   cudev     -> `find_package(OpenCV REQUIRED core cudev)` in the module
#   videoio   -> samples/app_video_mvs.cpp (mp4 decoding / encoding)
# calib3d/features2d/flann are kept as cheap insurance for viz's sample targets.
RUN --mount=type=cache,target=/opt/opencv-build \
    cmake -S /opt/opencv-src -B /opt/opencv-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/opencv \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv-contrib/modules \
        -DBUILD_LIST=core,imgproc,imgcodecs,highgui,calib3d,flann,features2d,viz,cudev,videoio \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_opencv_world=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
        -DCUDA_ARCH_PTX=${CUDA_ARCH_PTX} \
        -DWITH_CUDNN=OFF \
        -DWITH_CUFFT=OFF \
        -DWITH_CUBLAS=OFF \
        -DWITH_NVCUVID=OFF \
        -DWITH_NVCUVENC=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_IPP=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_ADE=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GTK=ON \
        -DWITH_QT=OFF \
        -DWITH_VTK=ON \
        -DENABLE_FAST_MATH=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
 && cmake --build /opt/opencv-build --parallel "$(nproc)" \
 && cmake --install /opt/opencv-build

# -----------------------------------------------------------------------------
# Stage 2: build CUMVS against the freshly built OpenCV
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS cumvs-builder

ARG CUDA_ARCH_BIN

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    OpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
    CMAKE_PREFIX_PATH=/opt/opencv \
    LD_LIBRARY_PATH=/opt/opencv/lib:/usr/local/cuda/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
        pkg-config \
        libgl1 \
        libglu1 \
        libgtk-3-0 \
        libvtk9.1 \
        libavcodec58 \
        libavformat58 \
        libavutil56 \
        libswscale5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv-builder /opt/opencv /opt/opencv

# The source and the build tree are kept in separate top-level directories so
# that bind-mounting the repository over /workspace/cumvs (the development
# workflow) can never shadow the prebuilt binaries in /opt/cumvs-build.
WORKDIR /opt/cumvs-src
COPY . .

RUN cmake -S /opt/cumvs-src -B /opt/cumvs-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_BIN} \
        -DOpenCV_DIR=${OpenCV_DIR} \
    && cmake --build /opt/cumvs-build --parallel "$(nproc)"

# Smoke test: exercises the full link chain (OpenCV shared libs, libcuda stub,
# VTK-backed viz).  `--help` prints the usage text and exits with 0.
RUN /opt/cumvs-build/samples/app_patch_match_mvs --help \
 && /opt/cumvs-build/samples/app_initialize_ETH3D --help \
 && /opt/cumvs-build/samples/app_video_mvs --help

# -----------------------------------------------------------------------------
# Stage 3: runtime image
#
# The `devel` CUDA base is used on purpose so that `nvcc` stays available and
# the CUMVS sources can be rebuilt inside the container without touching the
# host.  Swap in `nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}`
# if you only ever want to run the prebuilt binaries.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS runtime

ARG CUDA_ARCH_BIN

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/cumvs-build/samples:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/opencv/lib:/usr/local/cuda/lib64 \
    OpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
    CMAKE_PREFIX_PATH=/opt/opencv \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
        pkg-config \
        libgl1 \
        libglu1 \
        libgtk-3-0 \
        libvtk9.1 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-6 \
        xvfb \
        libavcodec58 \
        libavformat58 \
        libavutil56 \
        libswscale5 \
    && rm -rf /var/lib/apt/lists/* \
 && echo "/opt/opencv/lib" > /etc/ld.so.conf.d/opencv.conf \
 && ldconfig

COPY --from=opencv-builder /opt/opencv /opt/opencv
COPY --from=cumvs-builder /opt/cumvs-src /opt/cumvs-src
COPY --from=cumvs-builder /opt/cumvs-build /opt/cumvs-build

COPY docker/entrypoint.sh /usr/local/bin/cumvs-entrypoint
RUN chmod +x /usr/local/bin/cumvs-entrypoint

WORKDIR /opt/cumvs-src

ENTRYPOINT ["/usr/local/bin/cumvs-entrypoint"]
CMD ["/bin/bash"]
