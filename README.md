# cuda-multi-view-stereo

C++/CUDA library for Multi-View Stereo

---

## Description
cuda-multi-view-stereo (CUMVS) is a C++/CUDA library for Multi-View Stereo. This project currently supports depth map estimation using PatchMatch algorithms.

## Features

### Implementation of recent PatchMatch algorithms

This project aims to incorporate the essence of recent PatchMatch algorithms such as [ACMM](https://github.com/GhiXu/ACMM) while simplifying the implementation as much as possible.

### Performance optimization using GPU and CPU

The performance of CUDA kernels for propagation is optimized using shared memory, texture fetching, projective transformation approximations, etc. In addition, the performance of some processes performed on the CPU is optimized using multi-core parallelization and reduced disk access.

<div align="center">
<img src="https://github.com/fixstars/cuda-multi-view-stereo/wiki/images/runtime.png" width=1024>
</div>

### Support for debugging and visualization

The debug callback function allows you to check the status of the propagation during execution. In addition, you can instantly check the 3D reconstruction results using OpenCV's Viz module. Please check the [sample code](./samples/app_patch_match_mvs.cpp) for usage.

<div align="center">
<img src="https://github.com/fixstars/cuda-multi-view-stereo/wiki/images/demo.gif" width=1024><br/>
Visualization of propagation using debug callback (from left: reference image, depth map, normal map, cost map)
</div>

---

## Performance

### Settings
|Key|Value|
|---|---|
|CPU|Intel Core i7-13700K|
|GPU|NVIDIA GeForce RTX 3080|
|Maximum image size|3200 pixels for each dimension|

### Results on ETH3D High-res multi-view training dataset [2cm]

| method | Mean | courtyard | delivery_area | electro | facade | kicker | meadow | office | pipes | playground | relief | relief_2 | terrace | terrains |
|--------|------|-----------|---------------|---------|--------|--------|--------|--------|-------|------------|--------|----------|---------|----------|
| ACMM   | 78.9 | 87.5      | 83.7          | 86.8    | 70.8   | 77.5   | 66.4   | 64.9   | 70.1  | 72.1       | 85.4   | 84.9     | 89.9    | 85.5     |
| CUMVS  | 81.8 | 86.4      | 86.1          | 86.3    | 69.2   | 86.9   | 76.1   | 75.2   | 73    | 74.6       | 85.8   | 84.2     | 89.8    | 90.2     |

### Runtime (in second) on ETH3D High-res multi-view training dataset

| SCENE | Total   | courtyard | delivery_area | electro | facade | kicker | meadow | office | pipes | playground | relief | relief_2 | terrace | terrains |
|-------|---------|-----------|---------------|---------|--------|--------|--------|--------|-------|------------|--------|----------|---------|----------|
| ACMM  | 12277.7 | 1051.8    | 1014          | 1173.1  | 2471.1 | 908.6  | 337.2  | 595.7  | 233.5 | 1093.3     | 855.6  | 813.4    | 520.2   | 1210.2   |
| CUMVS | 2140.4  | 174.1     | 189.2         | 190.2   | 382.4  | 154.5  | 49.8   | 95.8   | 42.4  | 152.3      | 208.5  | 185.4    | 103.1   | 212.7    |

---

## Requirements
|Package Name|Minimum Requirements|Note|
|---|---|---|
|CMake|version >= 3.18||
|CUDA Toolkit|compute capability >= 6.0||
|C++ host compiler|supported by the installed CUDA Toolkit|nvcc rejects newer GCC/Clang (e.g. CUDA 12.9 accepts GCC <= 14)|
|OpenCV|version >= 4.6.0||
|OpenCV CUDA module|version >= 4.6.0|included in [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)|
|OpenCV Viz module|version >= 4.6.0|included in [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib)|
|VTK|version >= 9.0|needed for OpenCV Viz module|

---

## How to build

All samples include `cv::viz` (see `samples/draw.cpp`), so OpenCV has to provide the `viz` module, and that module needs VTK. `samples/CMakeLists.txt` additionally calls `find_package(OpenCV REQUIRED)` without a component list, i.e. the complete OpenCV module set has to be installed. An OpenCV built without VTK still configures fine, but every sample then fails to link with `undefined reference to cv::viz::Widget::~Widget()`. Check an installation with `opencv_version --verbose | grep -E "viz|VTK support"`.

`nvcc` only accepts host compilers up to a version tied to the toolkit. If the distribution default is newer (for example GCC 15 on Ubuntu 26.04 with CUDA 12.9), CMake fails while compiling `CMakeCUDACompilerId.cu`:

```
error: #error -- unsupported GNU version! gcc versions later than 14 are not supported!
```

In that case point CMake at a supported compiler (`-DCMAKE_CUDA_HOST_COMPILER=...`, with matching `CC`/`CXX`), or use the Docker image below.

```bash
$ git clone https://github.com/fixstars/cuda-multi-view-stereo.git
$ cd cuda-multi-view-stereo
$ mkdir build
$ cd build
# specify CUDA architecture if necessary (e.g. -DCMAKE_CUDA_ARCHITECTURES=86);
# -DOpenCV_DIR is only needed if OpenCV is not in a default search location
$ cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
$ make -j
```

### Building with Docker (no native toolchain required)

`Dockerfile` builds a self-contained image: CUDA 12.4.1, OpenCV 4.10.0 + opencv_contrib (`cudev` and the VTK-backed `viz`), VTK 9, and the samples with native SASS for compute capability 6.1. Per-host values (GPU architecture, dataset mount, container uid/gid) live in `.env`, which `docker-compose.yml` picks up automatically.

```bash
$ docker build -t cumvs .                  # ~15-25 min, OpenCV is built from source
$ docker run --gpus all -it --rm -v "$PWD":/workspace/cumvs cumvs
$ docker compose run --rm cumvs            # same, via docker-compose.yml
```

`CUDA_ARCH_BIN`/`CUDA_ARCH_PTX` in `.env` must match the GPU that runs the binaries (`61` Pascal/GTX 1080 Ti, `80` A100, `86` RTX 30xx, `89` L40/RTX 40xx, `90` H100) — a SASS build for one architecture does not run on another — and the image has to be rebuilt after changing them. Prebuilt binaries live in `/opt/cumvs-build` (also on `PATH`); rebuild the mounted working copy with:

```bash
$ cmake -S /workspace/cumvs -B /workspace/cumvs/build -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=61 -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
$ cmake --build /workspace/cumvs/build -j
```

### CMake options
|Option|Description|Default|
|---|---|---|
|CMAKE_CUDA_ARCHITECTURES|List of architectures to generate device code for|`52;61;72;75;86`|
|CMAKE_CUDA_HOST_COMPILER|Host compiler used by nvcc (must be supported by the CUDA Toolkit)|CMake's default C++ compiler|
|OpenCV_DIR|Directory containing `OpenCVConfig.cmake` (e.g. `/opt/opencv/lib/cmake/opencv4`)|system search path|
|CUMVS_BUILD_TOOLS|Build the developer tools in `tools/synthetic_scene` (synthetic scene generator + accuracy checker)|`OFF`|

Recent toolkits deprecate `52`, so the default list produces an nvcc warning; pass an explicit list matching your GPU (e.g. `-DCMAKE_CUDA_ARCHITECTURES=61`) to silence it and to skip the unused PTX variants.

## How to run
### samples

|Command|Description|
|---|---|
|`./samples/app_initialize_ETH3D scene-directory [options]`|Input data preparation for ETD3D dataset|
|`./samples/app_patch_match_mvs input-directory [options]`|Depth map estimation and 3D reconstruction|
|`./samples/app_video_mvs rig-directory [options]`|Depth maps and per-frame point clouds from a synchronised multi-camera video rig (`--calibration-file=...`)|

Use the `--help` or `-h` option for detailed information.

### Example of running CUMVS on ETH3D High-res multi-view data

Here we will use the "pipes" scene as an example.
Download `pipes_dslr_undistorted.7z` from [ETH3D datasets](https://www.eth3d.net/datasets) and unzip it.

Create input data using `app_initialize_ETH3D`.

```bash
./samples/app_initialize_ETH3D /path_to_eth3d/pipes_dslr_undistorted/pipes --output-directory=inputs/pipes
```

Test `app_patch_match_mvs`.
The default image size is large and time consuming, so we recommend trying a smaller size first.
The option `-d` specifies debug display.

```bash
./samples/app_patch_match_mvs inputs/pipes --max-image-size=800 -d
```

If there are no problems, run the application with your preferred options.
After execution, a point cloud file `output/point_cloud_dense.ply` will be generated.

### Running with Docker

The `cumvs` image already contains the built samples (on `PATH`), so running only needs the GPU and a mount of the working directory and data. Files the container writes (input dirs, `output/`, `build/`) are owned by your host user, not root.

```bash
# Docker directly: repo at /workspace/cumvs, dataset read-only at /Data
$ docker run --gpus all -it --rm \
    -v "$PWD":/workspace/cumvs \
    -v /path/to/datasets:/Data:ro \
    cumvs
```

or, with the same mounts configured in `.env` (`DATA_MOUNT_SOURCE`) and `docker-compose.yml`:

```bash
$ docker compose run --rm cumvs                # interactive shell
$ docker compose run --rm cumvs app_patch_match_mvs --help   # single command
```

Inside the container the samples are the prebuilt ones, e.g.:

```bash
# ETH3D example from above, with the dataset mounted at /Data
$ app_initialize_ETH3D /Data/pipes_dslr_undistorted/pipes --output-directory=/workspace/cumvs/inputs/pipes
$ app_patch_match_mvs /workspace/cumvs/inputs/pipes --max-image-size=800
```

Notes:

- Each container uses only the current CUDA device; on a multi-GPU host select one with `NVIDIA_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` (or `--gpus '"device=N"'`).
- `-d` opens GUI windows and therefore needs a display. On a headless machine run it under Xvfb: `xvfb-run -a app_patch_match_mvs ... -d`. With an X server, pass the socket and display through: `-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY`.
- Build artefacts written inside a container are owned by the uid/gid the entrypoint drops to (`CUMVS_UID`/`CUMVS_GID` in `.env`, else the owner of the mounted repo), so they stay editable on the host. A `build/` tree configured against a *different* OpenCV/CUDA (e.g. a native build) has to be removed before building inside the container.
- `docker compose` requires Compose v2.30+ for `gpus: all`; on older versions use `runtime: nvidia` with the `NVIDIA_VISIBLE_DEVICES` environment variable.
- A missing `DATA_MOUNT_SOURCE` path makes `docker compose run` fail at startup; comment out that volume (or point it at an existing directory) on hosts without the dataset.

---

## Future work
- Improving depth estimation accuracy using plane prior and adaptive patch deformation
- Speeding up [ETH3D Multi-View Evaluation Program](https://github.com/ETH3D/multi-view-evaluation) and incorporate it into this project
- Implementing processes after mesh reconstruction

---

## Author
The "adaskit Team"

The adaskit is an open-source project created by [Fixstars Corporation](https://www.fixstars.com/) and its subsidiary companies including [Fixstars Autonomous Technologies](https://at.fixstars.com/), aimed at contributing to the ADAS industry by developing high-performance implementations for algorithms with high computational cost.

---

## License
Apache License 2.0
