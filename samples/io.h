/*
Copyright 2025 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __IO_H__
#define __IO_H__

#include <opencv2/core.hpp>

#include <vector>

namespace cv
{

void makeDirectory(const String& path);

void saveImageInfo(const String& filename, const std::vector<String>& imageNames, InputArrayOfArrays cameraParams);
int loadImageInfo(const String& filename, std::vector<String>& imageNames, std::vector<Mat>& cameraParams);

void loadImages(const std::vector<String>& filenames, std::vector<Mat>& images, int flags);
void loadImagesAndCameras(const String& filenames, std::vector<Mat>& images, std::vector<Mat>& cameras, int maxImageSize = -1);

void saveGlobalPoses(const String& filename, InputArrayOfArrays Rs, InputArrayOfArrays ts);
void loadGlobalPoses(const String& filename, std::vector<Mat>& Rs, std::vector<Mat>& ts);

void saveViewIdSets(const String& filename, InputArrayOfArrays viewIdSets);
void loadViewIdSets(const String& filename, std::vector<Mat>& viewIdSets);

/** @brief The camera calibration of a multi-camera rig.
The layout follows the calibration files written by common rig calibration
tools: a "nb_camera" entry and one "camera_<i>" node per camera, each of them
holding a 3x3 intrinsic matrix, a distortion coefficient vector and a 4x4
camera pose matrix.
*/
struct CameraCalibration
{
	int nbCameras = 0;             //!< number of cameras in the rig
	std::vector<Mat> cameras;       //!< 3x3 intrinsic matrices (CV_64F)
	std::vector<Mat> distortions;   //!< distortion coefficient vectors (CV_64F)
	std::vector<Mat> poses;         //!< 4x4 camera pose matrices (CV_64F)
	std::vector<int> distortionTypes; //!< distortion model id stored in the file
	std::vector<int> cameraGroups;  //!< camera group id stored in the file
	std::vector<Size> imageSizes;   //!< image size the calibration was made for
};

/** @brief Loads a multi-camera rig calibration file.
@param filename Path of the calibration file.
@param calibration Output calibration.
@return The number of cameras read from the file.
*/
int loadCameraCalibration(const String& filename, CameraCalibration& calibration);

} // namespace cv

#endif // !__IO_H__
