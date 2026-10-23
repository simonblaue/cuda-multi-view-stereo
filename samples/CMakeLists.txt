cmake_minimum_required(VERSION 3.18)

project(samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

set(SRCS_COMMON io.cpp io.h draw.cpp draw.h yaml_lite.cpp yaml_lite.h rig_plot.cpp rig_plot.h)

add_executable(app_initialize_ETH3D app_initialize_ETH3D.cpp ${SRCS_COMMON})
target_link_libraries(app_initialize_ETH3D ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(app_patch_match_mvs app_patch_match_mvs.cpp ${SRCS_COMMON})
target_link_libraries(app_patch_match_mvs ${OpenCV_LIBS} OpenMP::OpenMP_CXX cuda_multi_view_stereo)

add_executable(app_video_mvs app_video_mvs.cpp ${SRCS_COMMON})
target_link_libraries(app_video_mvs ${OpenCV_LIBS} OpenMP::OpenMP_CXX cuda_multi_view_stereo)
