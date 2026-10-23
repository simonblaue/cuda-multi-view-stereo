cmake_minimum_required(VERSION 3.18)

project(CudaMultiViewStereo)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "52;61;72;75;86")
endif()

add_subdirectory(modules)
add_subdirectory(samples)

# Optional developer tools (synthetic scene generator + accuracy checker).
# Not built by default: enable with -DCUMVS_BUILD_TOOLS=ON.
option(CUMVS_BUILD_TOOLS "Build the developer tools in tools/" OFF)
if(CUMVS_BUILD_TOOLS)
	add_subdirectory(tools/synthetic_scene)
endif()
