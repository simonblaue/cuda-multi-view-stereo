cmake_minimum_required(VERSION 3.18)

project(synthetic_scene LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio viz)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O2 -Wall")
endif()

# make_scene       : renders a synthetic multi-view stereo rig and writes it in the
#                    exact FileStorage/PLY layout that app_initialize_ETH3D produces.
# make_video_scene : renders a synthetic synchronised multi-camera video sequence
#                    plus a rig calibration file for app_video_mvs.
# check_cloud      : compares the dense PLY against the ground-truth plane.
add_executable(make_scene make_scene.cpp)
target_link_libraries(make_scene ${OpenCV_LIBS})

add_executable(make_video_scene make_video_scene.cpp)
target_link_libraries(make_video_scene ${OpenCV_LIBS})

add_executable(check_cloud check_cloud.cpp)
target_link_libraries(check_cloud ${OpenCV_LIBS})
