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

#ifndef __RIG_PLOT_H__
#define __RIG_PLOT_H__

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace cv
{

/** @brief Description of a camera of a rig.
The world frame is the frame of the calibration file: the third column of the
camera-to-world rotation is the optical axis and the camera centre is the
translation of the pose.
*/
struct RigCamera
{
	Vec3d center;                //!< camera centre in world coordinates [m]
	Vec3d axis;                  //!< optical axis in world coordinates (unit vector)
	double horizontalFov = 0.0;  //!< horizontal field of view [deg]
};

/** @brief Measurements between two neighbouring cameras of a rig.
The distances are measured between the camera centres and the angles between
the optical axes, so they can be checked with a ruler and a protractor in the
real world.
*/
struct RigPairMeasurement
{
	int camera0 = 0;                 //!< index of the first camera
	int camera1 = 0;                 //!< index of the second camera
	double distance = 0.0;           //!< distance between the camera centres [m]
	double horizontal = 0.0;         //!< distance between the camera centres in the XZ plane [m]
	double height = 0.0;             //!< difference of the camera heights (world Y) [m]
	double axisAngle = 0.0;          //!< angle between the optical axes [deg]
	double yawAngle = 0.0;           //!< angle between the horizontal projections of the optical axes [deg]
	double triangulationAngle = 0.0; //!< angle between the optical axis of camera0 and the baseline [deg]
};

/** @brief Returns the optical axis of a camera in world coordinates.
@param rotation The camera-to-world rotation.
@return The optical axis (the third column of the rotation).
*/
Vec3d opticalAxis(const Mat& rotation);

/** @brief Measures the distances and the angles between the neighbouring cameras of a rig.
@param cameras The cameras of the rig.
@param pairs The pairs of neighbouring cameras (camera0 < camera1).
@return The measurements of the pairs.
*/
std::vector<RigPairMeasurement> measureRig(const std::vector<RigCamera>& cameras,
	const std::vector<std::pair<int, int>>& pairs);

/** @brief Draws the layout of a camera rig.
The drawing holds three coordinate plots of the world frame (the XZ, the XY and
the ZY plane), the optical axis and the field of view of every camera, the
baselines of the neighbouring cameras and two tables with the distances and
the angles of the rig.
@param cameras The cameras of the rig.
@param pairs The pairs of neighbouring cameras (camera0 < camera1).
@param measurements The measurements of the pairs.
@param title The title of the drawing.
@param subtitle The subtitle of the drawing.
@param image The output image (CV_8UC3).
*/
void drawRigLayout(const std::vector<RigCamera>& cameras, const std::vector<std::pair<int, int>>& pairs,
	const std::vector<RigPairMeasurement>& measurements, const std::string& title,
	const std::string& subtitle, Mat& image);

/** @brief Prints the distances and the angles between the neighbouring cameras of a rig.
@param cameras The cameras of the rig.
@param measurements The measurements of the pairs.
*/
void printRigMeasurements(const std::vector<RigCamera>& cameras, const std::vector<RigPairMeasurement>& measurements);

} // namespace cv

#endif // !__RIG_PLOT_H__
