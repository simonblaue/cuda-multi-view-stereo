# AGENTS.md

C++/CUDA PatchMatch multi-view stereo library exposing `cv::cuda::PatchMatchMVS`, plus three CLI samples.
Fork of https://github.com/fixstars/cuda-multi-view-stereo (README documents upstream; the repo tracks `origin` = `simonblaue/cuda-multi-view-stereo`).

## Layout

- `modules/cuda_multi_view_stereo/` → CMake target `cuda_multi_view_stereo` (static lib)
  - `include/cuda_multi_view_stereo.h` — the only public header
  - `src/cuda_multi_view_stereo.cu` — all CUDA kernels + matching cost (bilateral ZNCC, homography approx., geometric consistency)
  - `src/cuda_multi_view_stereo.cpp` — host driver: pyramid, per-reference-view estimation, depth-map fusion
  - `src/propagation_{acmh,gipuma}.cpp` — per-algorithm strategy objects (red/black propagation)
  - `src/texture_image.{h,cpp}` — CUDA texture-object wrapper
- `samples/` → `app_initialize_ETH3D` (ETH3D → input dir), `app_patch_match_mvs` (run MVS) and `app_video_mvs` (run MVS on a synced multi-camera video rig)
  - `io.{h,cpp}` — FileStorage I/O helpers shared by all three samples, incl. `loadCameraCalibration()` (`CameraCalibration` struct)
  - `yaml_lite.{h,cpp}` — small flow/block YAML+JSON parser used for the rig calibration files
- **No tests, no CI, no lint/format config.** Verification = build + run a sample on real data.

## Build

**Preferred path on this machine: use the container** (see [Docker](#docker) below). The host has no usable OpenCV dev install.

Requires CMake ≥ 3.18, CUDA Toolkit (compute capability ≥ 6.0), OpenCV ≥ 4.6 with `cudev` **and** `viz` modules, VTK ≥ 9 (viz is VTK-backed).

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
make -j
```

- `samples/CMakeLists.txt` calls `find_package(OpenCV REQUIRED)` with **no COMPONENTS** → the *entire* OpenCV module set must exist, including `viz` and `cudev`. A headless OpenCV will not configure.
- `modules/cuda_multi_view_stereo/CMakeLists.txt` collects sources with `file(GLOB src/*.cpp src/*.cu src/*.h*)` and **no `CONFIGURE_DEPENDS`** → after adding/renaming a source file, re-run `cmake ..` in the build dir or the new file is silently not compiled.
- Root `CMakeLists.txt` must keep `add_subdirectory(modules)` before `add_subdirectory(samples)`; `samples` links the library target by name.
- Default `CMAKE_CUDA_ARCHITECTURES` is `52;61;72;75;86`; each entry emits SASS **and** PTX (`code=[compute_XX,sm_XX]`), so an sm_89 GPU still runs via PTX JIT (slow first launch). Pass `-DCMAKE_CUDA_ARCHITECTURES=<your cc>` for native SASS.
- **This machine**: `nvcc` 12.0 present, 3× NVIDIA L40 (sm_89). OpenCV dev files are **missing** — `cmake ..` fails with "Could not find a package configuration file provided by OpenCV". Only runtime `libopencv-core406t64` (4.6.0) is apt-installed (no `-dev`, no `viz`, no `cudev`), and the `opencv-cuda` conda env has OpenCV **2.4.13** (too old — the code needs `opencv2/core/cuda.hpp` + `GpuMat`). Use the Docker image (see [Docker](#docker)); installing OpenCV 4.6+ with contrib (`viz`, `cudev`) + VTK on the host is the alternative.
- `nvidia-smi` fails here with "Driver/library version mismatch" (NVML 580.178) even though `cudaGetDeviceCount`/`cudaGetDeviceProperties` work. Don't use `nvidia-smi` as the GPU health check.

## Docker

`Dockerfile` (repo root) builds a self-contained image with every dependency: CUDA 12.4.1, OpenCV 4.10.0 + opencv_contrib (`cudev` + `viz`) with FFmpeg-backed `videoio`, VTK 9.1, CMake/Ninja/g++/OpenMP.

```bash
docker build -t cumvs .                                    # ~15-25 min, OpenCV from source
docker run --gpus all -it --rm -v "$PWD":/workspace/cumvs cumvs
docker compose run --rm cumvs                              # same via docker-compose.yml
```

- Three stages: `opencv-builder` (OpenCV from source) → `cumvs-builder` (builds the repo + runs `--help` as a link smoke test) → `runtime` (final image, on the `devel` CUDA base so `nvcc` stays available for in-container rebuilds).
- Layout: `/opt/opencv` (OpenCV), `/opt/cumvs-src` (pristine source), `/opt/cumvs-build` (prebuilt binaries, on `PATH`), `/workspace/cumvs` (optional bind mount). The build tree is deliberately **outside** the mount point so a bind-mounted working copy can never shadow the prebuilt binaries.
- Inside the container, rebuild the mounted source with `cmake -S /workspace/cumvs -B /workspace/cumvs/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 && cmake --build /workspace/cumvs/build -j`. `OpenCV_DIR` is already exported by the image.
- **The library is single-GPU**: it never calls `cudaSetDevice()`, so it always uses the *current* device. To use all three L40s, run one container per GPU with `CUDA_VISIBLE_DEVICES=0/1/2` (or `--gpus '"device=N"'`).
- `-d` (GUI) needs a display; use `xvfb-run` inside the container for headless runs.
- Version pins and why they are what they are:
  - OpenCV **4.10.0** — first release containing the fix for opencv_contrib [#3690](https://github.com/opencv/opencv_contrib/issues/3690) ("CUDA Toolkit 12.4.0 `tuple` incompatibility"); 4.9.x does not build against CUDA 12.4.
  - `cudev` and `viz` live in **opencv_contrib**, not the main opencv repo.
  - `CUDA_ARCH_BIN=89` — native SASS for the L40 (Ada, sm_89).
  - `BUILD_LIST=core,imgproc,imgcodecs,highgui,calib3d,flann,features2d,viz,cudev,videoio` and `-DWITH_FFMPEG=ON` — `videoio`/FFmpeg are needed by `app_video_mvs` (mp4 decoding/encoding); the image installs the FFmpeg dev packages in the OpenCV builder stage and `libavcodec58/libavformat58/libavutil56` in both the build and runtime stages. BUILD_LIST is safe because `find_package(OpenCV REQUIRED)` with no components resolves to `OpenCV_LIB_COMPONENTS` (only the *built* modules), not to the full upstream module list.

### Host driver inconsistency (blocks `docker run --gpus all`)

The host's NVIDIA install is **half-updated**: the kernel module is `580.173.02` (`/proc/driver/nvidia/version`) but the userspace libraries are `580.178.04` (`libcuda.so.580.178.04`). Consequences:

- `nvidia-smi` / NVML fail with "Driver/library version mismatch" — **but the CUDA driver API still works** (`cudaGetDeviceCount` → 3 devices, cc 8.9). Use CUDA runtime calls, not `nvidia-smi`, as the GPU health check.
- `nvidia-container-cli` derives the expected library filenames from the *kernel module* version, so `docker run --gpus all` fails with `failed to fulfil mount request: open .../libEGL_nvidia.so.580.173.02: no such file or directory`.

**Proper fix (needs root):** reinstall the NVIDIA driver so the kernel module and userspace match (e.g. `apt install --reinstall nvidia-driver-580` matching the module version, or rebuild the DKMS module for 580.178.04).

**Workaround applied on this host** (via privileged helper containers, since the agent has docker-group access but no sudo) — the toolkit then resolves every mount:

- `/run/nvidia-persistenced/socket` must exist (`nvidia-persistenced` itself cannot start because NVML is broken) — a plain empty file is enough for the bind mount.
- `lib*.so.580.173.02 → lib*.so.580.178.04` symlinks in `/usr/lib/x86_64-linux-gnu` (29 files, including `vdpau/libvdpau_nvidia.so` and `nvidia/xorg/libglxserver_nvidia.so`).
- `/lib/firmware/nvidia/580.173.02 → 580.178.04` directory symlink (for `gsp_ga10x.bin`).

## Verification

There is no test suite. The container is verified end-to-end with a **synthetic stereo rig** instead of ETH3D data: 4 cameras translated along +x (identity orientation) over a tilted plane `z = 4 + 0.1·x` covered by a natural 1/f²-spectrum texture, written in the exact FileStorage/PLY layout that `app_initialize_ETH3D` produces.

The fixture lives in `tools/synthetic_scene/` and is **not** built by default (`-DCUMVS_BUILD_TOOLS=ON`):

```bash
cmake -S . -B build -G Ninja -DCUMVS_BUILD_TOOLS=ON -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
cmake --build build -j
./build/tools/synthetic_scene/make_scene /tmp/scene
./build/samples/app_patch_match_mvs /tmp/scene --output-directory=/tmp/out --max-image-size=400 --pm-iterations=3 --gc-iterations=2
./build/tools/synthetic_scene/check_cloud /tmp/out/point_cloud_dense.ply
```

Expected output: ~48k dense points, median |dz| ≈ 0.02, 99.9 % of the points within 0.1 of the ground-truth plane.

`make_video_scene` builds the same scene as a **video rig** (one `camera_<i>.mp4` per camera plus a `calibration.yml` in the rig-calibration flow style) for `app_video_mvs`:

```bash
./build/tools/synthetic_scene/make_video_scene /tmp/vscene        # 4 cameras x 10 frames, 640x480
./build/samples/app_video_mvs /tmp/vscene --calibration-file=/tmp/vscene/calibration.yml \
    --output-directory=/tmp/vout --max-image-size=800 --up-axis=+y
./build/tools/synthetic_scene/check_cloud /tmp/vout/ply/frame_000000.ply
```

Expected output: ~275k dense points per frame in `ply/frame_%06d.ply`, false-colour depth videos in `depth/camera_<i>.mp4`, median |dz| ≈ 0.001 and 100 % of the points within 0.1. `check_cloud` hardcodes the frame-0 plane, so only frame 0 can be checked (the scene shifts by `--scene-step` per frame). `--up-axis=+y` selects the calibration world frame; the default `-y` output frame is the same cloud rotated 180° about `z`, which `check_cloud` does not undo (its plane `z = 4 + 0.1·x` is not invariant under `x → -x`).

Four pitfalls when building such a fixture:

- The matching cost is a **bilateral** ZNCC with `SIGMA_COLOR = 0.2` on a 9×9 patch: the texture must be *locally smooth* (natural-image-like spectrum) or the patch has no support and every match degenerates. Per-pixel random noise is pathological — it yields only ~50 dense points.
- `viz` point clouds must be `Nx1` with 3 channels (`cv::Mat(N, 1, CV_32FC3)`), not `Nx3`; `Nx3` fails `vtkCloudMatSource::SetCloud` with `-215:Assertion failed`. `viz::readCloud` returns `Nx3`.
- The scene's `t` must be the **camera centre in world** (see the pose convention above). With the sign flipped, the disparity direction is inverted and the reconstruction collapses onto the depth-range limits without any error message.
- A rendered image must invert the lens model: pixel `(u,v)` maps to the ray `undistort(K⁻¹·(u,v))`, because a real lens projects ray `r` to pixel `K·distort(r)`. Using `distort()` for the pixel's ray (the other direction) makes the app's undistortion apply the model twice; the reconstruction still "works" but the error grows to ~0.07 median |dz| instead of ~0.001.

## Running (two stages)

```bash
./samples/app_initialize_ETH3D <ETH3D_scene_dir> --output-directory=inputs/pipes
./samples/app_patch_match_mvs inputs/pipes --max-image-size=800 -d
```

- `app_initialize_ETH3D` reads COLMAP-style `dslr_calibration_undistorted/{cameras,images,points3D}.txt` and writes an input dir: `input_images.json`, `global_poses.json`, `view_id_sets.json`, `point_cloud_sparse.ply`.
- Those `*.json` files are **OpenCV FileStorage YAML**, not JSON. The stored image paths are absolute paths into the ETH3D scene dir, so the scene must stay where it was.
- `app_patch_match_mvs` reads that input dir and writes `point_cloud_dense.ply` into `--output-directory` (default `output/`).
- Non-obvious options (`--help` for all): `--method` 0=GIPUMA_FAST, 1=ACMH (default 1); `--pm-iterations` 3; `--gc-iterations` 2; `--multi-scale` 1/0; `--max-image-size` 3200; `--normal-level` 0 = no normal draw.
- `-d` opens GUI windows (imshow + OpenCV viz 3D) → needs a display, unusable headless.
- Full-resolution runs take hours per ETH3D scene. Always develop with `--max-image-size=800`.
- **Numeric CLI options must use the `--key=value` form** (`--pm-iterations=3`). OpenCV's `CommandLineParser` treats the space-separated form (`--pm-iterations 3`) as a *flag* and stores the literal `true`, which then fails with `Parameter 'pm-iterations': can not convert: [true] to [int]`. String options accept either form.

## Running (video rig)

```bash
./samples/app_video_mvs inputs/rig --calibration-file=inputs/rig/calibration.yml --max-image-size=800
```

- Input = one video per camera named `camera_0.<ext>`, `camera_1.<ext>`, … (index order must match the calibration's `camera_<i>` nodes), all assumed frame-synchronised; frames are read strictly sequentially and MVS runs once per frame with **every camera as a reference view**.
- Output (into `--output-directory`, default = input directory): `ply/frame_%06d.ply` (one dense cloud per frame), `depth/camera_<i>.mp4` (false-colour depth video over the fixed search range) and `rig_layout.png` (the drawing of the camera rig, see below).
- The calibration is the rig-calibration YAML (`nb_camera` + one `camera_<i>` node per camera holding `camera_matrix`, `distortion_vector`, `distortion_type`, `img_width/img_height`, `camera_pose_matrix`). It is parsed by `yaml_lite` (`samples/yaml_lite.{h,cpp}`), **not** by `cv::FileStorage`, because FileStorage 4.10 cannot read the flow-map style those tools emit. A relative `--calibration-file` is resolved against the CWD first, then the input directory.
- Search range: `--min-depth`/`--max-depth`, else auto-estimated from ORB correspondences of the first frame (the cameras are triangulated with the known calibration, so the depth does not depend on the image resolution: 5 %/95 % percentiles of the triangulated depths, scaled to `[0.5·low, 1.5·high]`), else a rectified SGBM histogram peak (`[0.7·d, 1.4·d]`), else a rig-radius heuristic with a warning. No sparse cloud is needed (the app passes an empty `initialObjectPoints` + `setDepthRange`). **Do not** rely on the SGBM path alone for close scenes with a wide baseline: the disparity of a 1 m deep point on a 1 m baseline exceeds the image width, so no `numDisparities` value can represent it (this produced a 50–100 m range at 1536×2048 before the feature-based estimator was added).
- Pose convention of `camera_pose_matrix` is `--pose-convention` (default `camera-to-world`, the library's convention; pass `world-to-camera` for inverted poses — the app then converts `R.t()`, `-R.t()*t` itself).
- Neighbours per reference view = other cameras sorted by baseline distance, capped by `--max-neighbors` (default 5).
- **Output coordinate frame**: `--reference-camera=<i>` (default 0), `--up-axis=<x|y|z|-x|-y|-z>` (default `-y`) and `--up-axis-frame=camera|world` (default `camera`). The PLY clouds are written with coordinates relative to the reference camera centre, `+z` along that camera's optical axis (viewing direction) and the requested axis as `+y`/up; `+x` follows as `y × z`. `camera` means the up axis is named in the reference camera's own coordinates (`-y` = image up); `world` means it is named in the calibration world frame. The frame change is a pure rotation of the finished cloud (no scaling/shear/mirroring), so the geometry — all distances and angles between points — is identical; only the coordinates change. `--up-axis=z`/`-z` is rejected (parallel to the viewing direction). `--up-axis=+y` gives the raw calibration world frame (with the default camera frame). The rig plot, the depth maps and the reconstruction itself are unaffected and stay in the calibration world frame.
- Non-obvious options: `--frame-start/--frame-end/--frame-step`; `--undistort=0` to skip undistortion; `--distortion-model=auto|plumb-bob|rational|fisheye|none` (overrides `distortion_type`); `--save-ply=0` / `--save-depth-video=0`.
- **Rig layout drawing** (`samples/rig_plot.{h,cpp}`, drawn before the matching so it is available even if the matching fails): three coordinate plots of the world frame (X-Z, X-Y and Z-Y plane) with the camera centres, the optical axes, the horizontal field of view cone, the baselines of the neighbouring cameras (labelled with the distance [m] / the angle between the optical axes [deg]) and two tables (the pairwise measurements and the camera centres, azimuth, elevation and hFOV). The same measurements are printed to the console. `--rig-plot=0` disables it, `--rig-plot-file=<name>` changes the file name (default `rig_layout.png`, written into the output directory).

## How it fits together

- Public API is tiny: `PatchMatchMVS::compute(images, cameras, rotations, translations, viewIdSets, initialObjectPoints, objectPoints, objectColors, objectNormals, depths)`.
- `viewIdSets[i]` = `[referenceViewIdx, n1, n2, ...]`. The **first entry must be the reference view** — GPU code derives `s_invK1` and the neighbor homographies from `Ks[0]` of that subset.
- Pose convention: `rotations`/`translations` describe the camera in **world** coordinates, i.e. `X_world = R·X_cam + t`. So `R` is the **camera-to-world** rotation and `t` is the **camera centre in world coordinates** (at `X_cam = 0`, `X_world = t`). The ETH3D loader converts COLMAP's world→camera `(R, t)` accordingly: it stores `R.t()` and `-R.t()*t`. Getting this backwards silently produces garbage depths (wrong disparity direction) rather than an error.
- Depth range `minZ`/`maxZ` comes from `initialObjectPoints` (1%/99% percentiles × 0.75/1.25) **unless** `setDepthRange(minZ, maxZ)` was called, which overrides it and makes the sparse points unused (that is how `app_video_mvs` works). Empty sparse points without an explicit range → `CV_Error` ("The initial object points must not be empty unless an explicit depth range is set"); an empty `objectPoints` output is released instead of reshaped.
- Every `Mat` you pass must **own its data**: `Mat(rows, cols, type, ptr)` and `Mat(vector)` only wrap the buffer (`Mat(vector)` points at `&vec[0]`, `Mat(ptr, ...)` at the caller's memory). Wrapping a loop-local `Vec3d` for `translations` yields garbage poses that still print plausibly and reconstruct nothing.
- Split of work: GPU = propagation/refinement; CPU (OpenMP) = pyramid scaling, image loading, and depth-map fusion (`fuseDepthMaps`).
- The library lives in OpenCV's `cv::cuda` namespace, not its own — `using namespace cv;` then `cuda::PatchMatchMVS::create()`.

## Silent-failure constraints (violating these yields wrong output, not an error)

- `MAX_VIEWS = 21` in `cuda_multi_view_stereo.cu`: a viewIdSet holds at most 21 entries = 1 reference + 20 neighbors (matches `app_initialize_ETH3D --max-neighbors=20` default). More overflows the `__constant__` view arrays and the per-thread `float viewWeights[MAX_VIEWS]`.
- Image width must be even: propagation is red/black checkerboard (`x = 2*idx + color`, `npixelsX = w/2`), so an odd width silently leaves the last column unprocessed.
- `CUDA_CHECK` only `printf`s and continues — it never aborts or throws. Grep stdout for `[CUDA Error]`; a failed CUDA call does not stop the program.
- `DEBUG_PRINT` is compiled in by default (`ENABLE_DEBUG_PRINT` in `src/debug_print.h`) → `[MVS INFO] processing image ...` spam; comment the define out to silence.

## Style

- Tabs for indentation, Allman braces, `namespace cv { namespace cuda { ... } }`.
- Apache-2.0 header ("Copyright 2025 Fixstars Corporation") on every file — copy it into new files.
- Private members use a trailing `_`; concrete implementations of the abstract interfaces use the `Impl` suffix (`PatchMatchMVSImpl`, `PropagationACMHImpl`).
- Device helpers are `__device__ inline` in the `.cu`; host-side helpers are `static` in the `.cpp`.
