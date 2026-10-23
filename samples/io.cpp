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

#include "io.h"
#include "yaml_lite.h"

#include <filesystem>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace cv
{

static double getImageScale(Size imgSize, int maxImageSize)
{
	const int size = std::max(imgSize.width, imgSize.height);
	if (size > maxImageSize)
		return 1. * maxImageSize / size;
	return 1;
}

static Mat scaleImage(const Mat& src, double scale)
{
	Mat dst;
	const int interpolation = scale < 1.0 ? INTER_AREA : INTER_LINEAR;
	resize(src, dst, Size(), scale, scale, interpolation);
	return dst;
}

static Mat scaleCameraParams(const Mat& src, double scale)
{
	Mat dst = src.clone();
	dst(Range(0, 2), Range(0, 3)) *= scale;
	return dst;
}

void makeDirectory(const String& path)
{
	std::filesystem::create_directories(path);
}

void saveImageInfo(const String& filename, const std::vector<String>& imageNames, InputArrayOfArrays _cameraParams)
{
	const int nimages = static_cast<int>(imageNames.size());

	std::vector<Mat> cameraParams;
	_cameraParams.getMatVector(cameraParams);

	FileStorage fs(filename, FileStorage::WRITE);
	fs << "num_images" << nimages;
	fs << "input_images" << "[";
	for (int i = 0; i < nimages; i++)
	{
		fs << "{:";
		fs << "id" << i;
		fs << "filename" << imageNames[i];
		fs << "K" << cameraParams[i].reshape(1, 3);
		fs << "}";
	}
	fs << "]";
}

int loadImageInfo(const String& filename, std::vector<String>& imageNames, std::vector<Mat>& cameraParams)
{
	FileStorage fs(filename, FileStorage::READ);
	CV_Assert(fs.isOpened());

	int nimages = 0;
	fs["num_images"] >> nimages;

	imageNames.clear();
	imageNames.reserve(nimages);

	cameraParams.clear();
	cameraParams.reserve(nimages);

	for (const auto& node : fs["input_images"])
	{
		String imageName;
		Mat K;
		node["filename"] >> imageName;
		node["K"] >> K;
		imageNames.push_back(imageName);
		cameraParams.push_back(K);
	}

	return nimages;
}

void loadImages(const std::vector<String>& filenames, std::vector<Mat>& images, int flags)
{
	const int nimages = static_cast<int>(filenames.size());

	images.resize(nimages);

#pragma omp parallel for
	for (int i = 0; i < nimages; i++)
		images[i] = imread(filenames[i], flags);
}

void loadImagesAndCameras(const String& filenames, std::vector<Mat>& images, std::vector<Mat>& cameras, int maxImageSize)
{
	std::vector<std::string> imageNames;
	const int nimages = loadImageInfo(filenames, imageNames, cameras);

	images.resize(nimages);

#pragma omp parallel for
	for (int i = 0; i < nimages; i++)
	{
		const Mat image = imread(imageNames[i], IMREAD_COLOR);
		const double imageScale = maxImageSize > 0 ? getImageScale(image.size(), maxImageSize) : 0;
		images[i] = imageScale < 1 ? scaleImage(image, imageScale) : image;
		cameras[i] = imageScale < 1 ? scaleCameraParams(cameras[i], imageScale) : cameras[i];
	}
}

void saveGlobalPoses(const String& filename, InputArrayOfArrays _Rs, InputArrayOfArrays _ts)
{
	std::vector<Mat> Rs, ts;
	_Rs.getMatVector(Rs);
	_ts.getMatVector(ts);

	const int nimages = static_cast<int>(Rs.size());

	FileStorage fs(filename, FileStorage::WRITE);
	fs << "num_global_poses" << nimages;
	fs << "global_poses" << "[";
	for (int i = 0; i < nimages; i++)
	{
		fs << "{:";
		fs << "id" << i;
		fs << "R" << Rs[i].reshape(1, 3);
		fs << "t" << ts[i].reshape(1, 3);
		fs << "}";
	}
	fs << "]";
}

void loadGlobalPoses(const String& filename, std::vector<Mat>& Rs, std::vector<Mat>& ts)
{
	FileStorage fs(filename, FileStorage::READ);
	CV_Assert(fs.isOpened());

	int nimages = 0;
	fs["num_global_poses"] >> nimages;

	Rs.clear();
	Rs.reserve(nimages);

	ts.clear();
	ts.reserve(nimages);

	for (const auto& node : fs["global_poses"])
	{
		Mat R;
		Mat t;
		node["R"] >> R;
		node["t"] >> t;
		Rs.push_back(R);
		ts.push_back(t);
	}
}

void saveViewIdSets(const String& filename, InputArrayOfArrays _viewIdSets)
{
	std::vector<Mat> viewIdSets;
	_viewIdSets.getMatVector(viewIdSets);

	const int nreferences = static_cast<int>(viewIdSets.size());

	FileStorage fs(filename, FileStorage::WRITE);
	fs << "num_reference_images" << nreferences;
	fs << "view_id_sets" << "[";
	for (int i = 0; i < nreferences; i++)
	{
		const auto& viewIds = viewIdSets[i];

		fs << "{:";
		fs << "id" << viewIds.at<int>(0);
		fs << "view_id_set" << viewIds;
		fs << "}";
	}
	fs << "]";
}

void loadViewIdSets(const String& filename, std::vector<Mat>& viewIdSets)
{
	FileStorage fs(filename, FileStorage::READ);
	CV_Assert(fs.isOpened());

	int nreferences = 0;
	fs["num_reference_images"] >> nreferences;

	viewIdSets.clear();
	viewIdSets.reserve(nreferences);

	for (const auto& node : fs["view_id_sets"])
	{
		Mat viewIds;
		node["view_id_set"] >> viewIds;
		viewIdSets.push_back(viewIds);
	}
}

int loadCameraCalibration(const String& filename, CameraCalibration& calibration)
{
	// the OpenCV file storage cannot read the flow mappings that hold a
	// matrix, which is the layout of the calibration files, so the file is
	// parsed here
	const YamlNode root = parseYamlFile(filename);

	const int nbCameras = root.getInt("nb_camera", 0);
	const std::string message = format("The calibration file '%s' must contain a positive 'nb_camera' entry", filename.c_str());
	if (nbCameras <= 0)
		CV_Error(Error::StsBadArg, message.c_str());

	calibration.nbCameras = nbCameras;
	calibration.cameras.clear();
	calibration.distortions.clear();
	calibration.poses.clear();
	calibration.distortionTypes.clear();
	calibration.cameraGroups.clear();
	calibration.imageSizes.clear();

	for (int i = 0; i < nbCameras; i++)
	{
		const YamlNode* node = root.find(format("camera_%d", i));

		const std::string missing = format("The calibration file '%s' has no 'camera_%d' entry", filename.c_str(), i);
		if (node == nullptr)
			CV_Error(Error::StsBadArg, missing.c_str());

		Mat K = node->getMatrix("camera_matrix");
		Mat pose = node->getMatrix("camera_pose_matrix");
		Mat D = node->getMatrix("distortion_vector");

		const std::string invalid = format("The 'camera_%d' entry of '%s' is malformed", i, filename.c_str());
		if (K.rows != 3 || K.cols != 3)
			CV_Error(Error::StsBadArg, invalid.c_str());
		if (pose.rows != 4 || pose.cols != 4)
			CV_Error(Error::StsBadArg, invalid.c_str());

		K.convertTo(K, CV_64F);
		pose.convertTo(pose, CV_64F);
		if (!D.empty())
			D.convertTo(D, CV_64F);
		else
			D = Mat::zeros(1, 5, CV_64F);

		const int distortionType = node->getInt("distortion_type", 0);
		const int cameraGroup = node->getInt("camera_group", 0);
		const int width = node->getInt("img_width", 0);
		const int height = node->getInt("img_height", 0);

		calibration.cameras.push_back(K);
		calibration.distortions.push_back(D);
		calibration.poses.push_back(pose);
		calibration.distortionTypes.push_back(distortionType);
		calibration.cameraGroups.push_back(cameraGroup);
		calibration.imageSizes.push_back(Size(width, height));
	}

	return nbCameras;
}

} // namespace cv
