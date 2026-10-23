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

/*
Runs PatchMatch multi-view stereo on a synchronised multi-camera video sequence.

The input directory holds one video per camera (camera_0.mp4, camera_1.mp4, ...)
and a calibration file describing the intrinsics, the distortion and the pose of
every camera.  For every frame of the sequence the dense point cloud is computed
on the GPU and written as a PLY file, and the depth map of every reference view
is written as a video.

    app_video_mvs <input-directory> --calibration-file=<file>

Output layout (relative to the output directory):

    ply/frame_000000.ply     dense point cloud of frame 0
    depth/camera_0.mp4       depth map of camera 0 (false colour)
    depth/camera_1.mp4       depth map of camera 1 (false colour)

The point clouds are written in an output frame whose origin is the centre of the
reference camera (--reference-camera) and whose +z axis is the optical axis of
that camera (its viewing direction).  The up direction is chosen with --up-axis
and --up-axis-frame.  The frame change is a pure rotation, so the geometry of the
cloud is unchanged.
*/

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <omp.h>

#include <cuda_multi_view_stereo.h>

#include "io.h"
#include "draw.h"
#include "rig_plot.h"

static std::string keys =
"{ @input-directory | <none> | directory containing the camera_N.<ext> video files.                        }"
"{ calibration-file | calibration.yml | camera calibration file.                                          }"
"{ output-directory | <none> | output directory (default: the input directory).                          }"
"{ method           |      1 | PatchMatch method (0:GIPUMA_FAST 1:ACMH).                                   }"
"{ pm-iterations    |      3 | number of innier PatchMatch iterations.                                     }"
"{ gc-iterations    |      2 | number of outer PatchMatch iterations performed with geometric consistency. }"
"{ multi-scale      |      1 | whether to use multi scale.                                                 }"
"{ max-image-size   |    800 | maximum width or height of processed images.                                }"
"{ max-neighbors    |      5 | maximum number of neighbor views used per reference view.                    }"
"{ reference-camera |      0 | camera that defines the origin and the viewing direction of the output frame. }"
"{ up-axis          |     -y | up axis of the output frame (x, y, z, -x, -y, -z).                             }"
"{ up-axis-frame    | camera | frame the up axis is given in (camera: reference camera, world: calibration). }"
"{ min-depth        |     -1 | minimum depth of the search range (auto-estimated when negative).            }"
"{ max-depth        |     -1 | maximum depth of the search range (auto-estimated when negative).            }"
"{ frame-start      |      0 | index of the first frame to process.                                          }"
"{ frame-end        |     -1 | index of the last frame to process (-1: up to the end).                       }"
"{ frame-step       |      1 | frame step.                                                                  }"
"{ undistort        |      1 | whether the input images are undistorted before the matching.                }"
"{ distortion-model |  auto | distortion model (auto, plumb-bob, rational, fisheye or none).             }"
"{ pose-convention  | camera-to-world | pose convention of camera_pose_matrix (camera-to-world or world-to-camera). }"
"{ save-ply         |      1 | whether the dense point clouds are written.                                   }"
"{ save-depth-video |      1 | whether the depth map videos are written.                                     }"
"{ rig-plot         |      1 | whether the layout of the camera rig is drawn (distances and angles).        }"
"{ rig-plot-file    | rig_layout.png | name of the drawing of the camera rig (in the output directory).     }"
"{ num-threads      |     -1 | number of threads for CPU processing (-1:auto detect).                      }"
"{ debug d          |        | enable debug view.                                                          }"
"{ help  h          |        | print help message.                                                         }";

namespace cv
{

/** @brief Throws an exception with the given message if the condition is false.
@param condition The condition to check.
@param message The message of the exception.
*/
static void checkTrue(bool condition, const std::string& message)
{
	if (!condition)
		CV_Error(Error::StsBadArg, message.c_str());
}

static void debugCallBack(InputArray _image, InputArray _depths, InputArray _normals, InputArray _costs, const String& title)
{
	auto getMat = [](InputArray arr) -> Mat
	{
		Mat mat;
		if (arr.kind() == _InputArray::MAT)
			mat = arr.getMat();
		else if (arr.kind() == _InputArray::CUDA_GPU_MAT)
			arr.getGpuMat().download(mat);
		else
			CV_Error(Error::StsBadArg, "Unsupported InputArray kind");
		return mat;
	};

	const Mat image = getMat(_image);
	const Mat depths = getMat(_depths);
	const Mat normals = getMat(_normals);
	const Mat costs = getMat(_costs);

	const Mat depthColored = colored(depths);
	const Mat normalColored = coloredNormal(normals);
	const Mat costsColored = colored(costs);

	rectangle(depthColored, Rect(10, 10, 500, 70), Scalar::all(255), -1);
	putText(depthColored, format("%s", title.c_str()), Point(20, 40), 1, 2, Scalar(0, 0, 255), 2);
	putText(depthColored, format("Total cost: %.1f", sum(costs)[0]), Point(20, 70), 1, 2, Scalar(0, 0, 255), 2);

	imshow("input image", image);
	imshow("depth map", depthColored);
	imshow("normal map", normalColored);
	imshow("cost map", costsColored);
	waitKey(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Distortion
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum DistortionModel
{
	DISTORTION_NONE,
	DISTORTION_PLUMB_BOB,
	DISTORTION_RATIONAL,
	DISTORTION_FISHEYE
};

static DistortionModel resolveDistortionModel(int distortionType, int ncoeffs, const String& requested)
{
	if (requested == "none")
		return DISTORTION_NONE;
	if (requested == "plumb-bob")
		return DISTORTION_PLUMB_BOB;
	if (requested == "rational")
		return DISTORTION_RATIONAL;
	if (requested == "fisheye")
		return DISTORTION_FISHEYE;

	checkTrue(requested == "auto", "The distortion model must be one of auto, plumb-bob, rational, fisheye or none");

	// the calibration files store the model id: 0 = plumb-bob, 1 = rational
	// polynomial, 2 = fisheye.  Unknown ids fall back to the number of
	// coefficients.
	if (distortionType == 0)
		return DISTORTION_PLUMB_BOB;
	if (distortionType == 1)
		return DISTORTION_RATIONAL;
	if (distortionType == 2)
		return DISTORTION_FISHEYE;

	if (ncoeffs == 0)
		return DISTORTION_NONE;
	if (ncoeffs == 4)
		return DISTORTION_FISHEYE;
	if (ncoeffs == 5)
		return DISTORTION_PLUMB_BOB;
	if (ncoeffs == 8)
		return DISTORTION_RATIONAL;

	CV_Error(Error::StsBadArg, format("Cannot guess the distortion model from %d coefficients", ncoeffs).c_str());
}

static String distortionModelName(DistortionModel model)
{
	switch (model)
	{
	case DISTORTION_NONE: return "none";
	case DISTORTION_PLUMB_BOB: return "plumb-bob";
	case DISTORTION_RATIONAL: return "rational";
	case DISTORTION_FISHEYE: return "fisheye";
	}
	return "unknown";
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Poses
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static Matx33d toMatx33(const Mat& m)
{
	CV_Assert(m.rows == 3 && m.cols == 3 && m.type() == CV_64F && m.isContinuous());
	return Matx33d(m.ptr<double>());
}

/** @brief Converts the pose matrices of the calibration file into the convention used by the library.
The library expects X_world = R * X_cam + t, i.e. R is the camera-to-world
rotation and t is the camera centre in world coordinates.
@param poses The 4x4 pose matrices of the calibration file.
@param worldToCamera If true, the poses are interpreted as world-to-camera.
@param rotations Output camera-to-world rotations.
@param translations Output camera centres in world coordinates.
*/
static void extractPoses(const std::vector<Mat>& poses, bool worldToCamera,
	std::vector<Mat>& rotations, std::vector<Mat>& translations, std::vector<Vec3d>& centers)
{
	rotations.clear();
	translations.clear();
	centers.clear();
	rotations.reserve(poses.size());
	translations.reserve(poses.size());
	centers.reserve(poses.size());

	for (const auto& pose : poses)
	{
		// the rotation is the upper left 3x3 block and the translation is the
		// fourth column of the 4x4 matrix
		const Matx33d R = toMatx33(pose(Rect(0, 0, 3, 3)).clone());
		const Vec3d t(pose(Rect(3, 0, 1, 3)).clone());

		Matx33d Rw;
		Vec3d cw;
		if (worldToCamera)
		{
			// X_cam = R * X_world + t  =>  X_world = R^T * X_cam - R^T * t
			Rw = R.t();
			cw = -R.t() * t;
		}
		else
		{
			Rw = R;
			cw = t;
		}

		rotations.push_back(Mat(Rw));
		// Mat(data, ...) only wraps the buffer, so the values must be copied:
		// cw is a loop local and the library expects a 3x1 vector
		translations.push_back((Mat_<double>(3, 1) << cw[0], cw[1], cw[2]));
		centers.push_back(cw);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Videos
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static String findVideoFile(const String& directory, int cameraIndex)
{
	static const char* extensions[] = { ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".MP4", ".AVI", ".MOV", ".MKV" };

	const String prefix = format("camera_%d", cameraIndex);
	for (const auto* extension : extensions)
	{
		const String path = directory + "/" + prefix + extension;
		if (std::filesystem::exists(path))
			return path;
	}

	// fall back to any file whose name starts with the camera prefix
	for (const auto& entry : std::filesystem::directory_iterator(directory))
	{
		if (!entry.is_regular_file())
			continue;
		const String name = entry.path().filename().string();
		if (name.rfind(prefix, 0) == 0)
			return entry.path().string();
	}

	return String();
}

static int openVideoWriter(const String& path, double fps, const Size& size, VideoWriter& writer)
{
	const char* fourccs[] = { "mp4v", "avc1", "MJPG" };

	for (const auto* fourcc : fourccs)
	{
		writer.open(path, cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]), fps, size, true);
		if (writer.isOpened())
			return 0;
		writer.release();
	}

	return -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Depth visualisation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static Mat colorizeDepth(const Mat& depth, double minZ, double maxZ)
{
	Mat dst;
	if (depth.empty() || maxZ <= minZ)
		return dst;

	Mat gray;
	depth.convertTo(gray, CV_32F);
	gray = (gray - static_cast<float>(minZ)) * (255.f / static_cast<float>(maxZ - minZ));
	gray.setTo(0.f, gray < 0.f);
	gray.setTo(255.f, gray > 255.f);
	gray.convertTo(dst, CV_8U);

	Mat color;
	applyColorMap(dst, color, COLORMAP_TURBO);
	return color;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Depth range estimation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct DepthRange
{
	bool valid = false;
	double minZ = 0.0;
	double maxZ = 0.0;
	std::string detail;
};

/** @brief Robust percentile of a sorted vector.
@param sorted The sorted values.
@param q The quantile in [0, 1].
*/
static double percentile(const std::vector<double>& sorted, double q)
{
	if (sorted.empty())
		return 0.0;
	const double pos = q * static_cast<double>(sorted.size() - 1);
	const size_t lo = static_cast<size_t>(std::floor(pos));
	const size_t hi = std::min(lo + 1, sorted.size() - 1);
	return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

/** @brief Estimates the depth range of the scene by triangulating feature
correspondences.
The cameras all point at the same scene, so the depth of every triangulated
correspondence is a sample of the scene depth.  Compared to a rectified stereo
match this does not suffer from the disparity range of the matcher: with a wide
baseline and a close scene the disparity can exceed the image width, so a stereo
matcher cannot represent it at any resolution.  The 5% and 95% percentiles of the
triangulated depths bound the scene without being sensitive to mismatches.
*/
static DepthRange estimateDepthRangeFromFeatures(const std::vector<Mat>& images,
	const std::vector<Mat>& cameras, const std::vector<Mat>& rotations,
	const std::vector<Vec3d>& centers)
{
	DepthRange result;

	const int ncameras = static_cast<int>(images.size());

	// the feature detector is limited to 5000 keypoints per image; that is
	// plenty for a depth histogram and keeps the detection fast at full
	// resolution
	const Ptr<ORB> orb = ORB::create(5000);
	std::vector<std::vector<KeyPoint>> keypoints(ncameras);
	std::vector<Mat> descriptors(ncameras);
	for (int i = 0; i < ncameras; i++)
	{
		Mat gray;
		if (images[i].channels() == 3)
			cvtColor(images[i], gray, COLOR_BGR2GRAY);
		else
			gray = images[i];
		orb->detectAndCompute(gray, noArray(), keypoints[i], descriptors[i]);
	}

	std::vector<Matx33d> invK(ncameras);
	for (int i = 0; i < ncameras; i++)
		invK[i] = toMatx33(cameras[i]).inv();

	std::vector<double> depths;
	const Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
	int npairs = 0;

	for (int i = 0; i < ncameras; i++)
	{
		for (int j = i + 1; j < ncameras; j++)
		{
			if (descriptors[i].empty() || descriptors[j].empty())
				continue;

			// X_cam_j = Rrel * X_cam_i + Trel
			const Matx33d Rrel = toMatx33(rotations[j]).t() * toMatx33(rotations[i]);
			const Vec3d Trel = toMatx33(rotations[j]).t() * (centers[i] - centers[j]);
			if (cv::norm(Trel) < 1e-9)
				continue;

			std::vector<std::vector<DMatch>> knn;
			matcher->knnMatch(descriptors[i], descriptors[j], knn, 2);

			int count = 0;
			for (const auto& matches : knn)
			{
				// the ratio test rejects ambiguous (repetitive) texture
				if (matches.size() < 2 || matches[0].distance > 0.75 * matches[1].distance)
					continue;

				const Point2f pi = keypoints[i][matches[0].queryIdx].pt;
				const Point2f pj = keypoints[j][matches[0].trainIdx].pt;
				const Vec3d xi = invK[i] * Vec3d(pi.x, pi.y, 1.0);
				const Vec3d xj = invK[j] * Vec3d(pj.x, pj.y, 1.0);

				// the point is lambda * xi in camera i and mu * xj in camera j,
				// i.e. lambda * (Rrel * xi) - mu * xj = -Trel; solved in the
				// least squares sense for the two unknown depths
				const Vec3d a = Rrel * xi;
				const double aa = a.dot(a);
				const double ab = a.dot(xj);
				const double bb = xj.dot(xj);
				const double at = a.dot(Trel);
				const double bt = xj.dot(Trel);
				const double det = aa * bb - ab * ab;
				if (std::abs(det) < 1e-12)
					continue;

				const double lambda = (-at * bb + ab * bt) / det;
				const double mu = (aa * bt - ab * at) / det;
				if (!(lambda > 0.0) || !(mu > 0.0))
					continue;

				// the reprojection of the triangulated point into the other
				// image is the residual of the correspondence
				const Vec3d Xj = Rrel * (lambda * xi) + Trel;
				const Vec3d XjProj = toMatx33(cameras[j]) * Xj;
				if (XjProj[2] <= 1e-9)
					continue;
				const double dx = XjProj[0] / XjProj[2] - pj.x;
				const double dy = XjProj[1] / XjProj[2] - pj.y;
				if (dx * dx + dy * dy > 2.0 * 2.0)
					continue;

				if (!(lambda > 0.05) || !(lambda < 1000.0))
					continue;

				depths.push_back(lambda);
				count++;
			}

			if (count > 0)
				npairs++;
		}
	}

	if (static_cast<int>(depths.size()) < 20 || npairs < 1)
	{
		result.detail = format("the feature matching found only %d correspondences on %d camera pairs",
			static_cast<int>(depths.size()), npairs);
		return result;
	}

	std::sort(depths.begin(), depths.end());
	const double median = percentile(depths, 0.5);
	const double low = percentile(depths, 0.05);
	const double high = percentile(depths, 0.95);

	result.valid = true;
	result.minZ = 0.5 * low;
	result.maxZ = 1.5 * high;
	// a scene of a single distance has no spread, so the range is widened
	if (result.maxZ < 2.0 * result.minZ)
		result.maxZ = 2.0 * result.minZ;
	result.detail = format("%d feature correspondences on %d camera pairs, median depth %.3f, "
		"5%%..95%% %.3f..%.3f", static_cast<int>(depths.size()), npairs, median, low, high);
	return result;
}

/** @brief Estimates the depth range of the scene with a rectified stereo match.
Every pair of cameras is rectified and matched with SGBM; the pair with the
sharpest disparity histogram gives the dominant disparity and therefore the
dominant depth of the scene.
*/
static DepthRange estimateDepthRange(const std::vector<Mat>& images, const std::vector<Mat>& cameras,
	const std::vector<Mat>& rotations, const std::vector<Vec3d>& centers)
{
	DepthRange result;

	const int ncameras = static_cast<int>(images.size());
	const int disparityRanges[] = { 64, 128, 256 };

	double bestScore = 0.0;
	double bestMode = 0.0;
	double bestBaseline = 0.0;
	double bestFocal = 0.0;
	int bestPair0 = -1;
	int bestPair1 = -1;

	for (int i = 0; i < ncameras; i++)
	{
		for (int j = i + 1; j < ncameras; j++)
		{
			if (images[i].size() != images[j].size())
				continue;

			const Matx33d Ri = toMatx33(rotations[i]);
			const Matx33d Rj = toMatx33(rotations[j]);
			// X_cam_j = R_rel * X_cam_i + T_rel
			const Matx33d Rrel = Rj.t() * Ri;
			const Vec3d Trel = Rj.t() * (centers[i] - centers[j]);

			const double baseline = std::sqrt(Trel[0] * Trel[0] + Trel[1] * Trel[1] + Trel[2] * Trel[2]);
			if (baseline < 1e-9)
				continue;

			// stereoRectify may modify the rotation and the translation, so it
			// gets writable copies of them
			const Mat RrelMat(Rrel);
			Mat TrelMat = (Mat_<double>(3, 1) << Trel[0], Trel[1], Trel[2]);

			Mat R1, R2, P1, P2, Q;
			// alpha = 0 keeps only valid pixels, so the rectified images do not
			// get black borders
			cv::stereoRectify(cameras[i], Mat(), cameras[j], Mat(), images[i].size(), RrelMat, TrelMat,
				R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0.0);

			const double focal = P1.at<double>(0, 0);
			if (focal < 1.0)
				continue;

			Mat map1i, map2i, map1j, map2j;
			cv::initUndistortRectifyMap(cameras[i], Mat(), R1, P1, images[i].size(), CV_32FC1, map1i, map2i);
			cv::initUndistortRectifyMap(cameras[j], Mat(), R2, P2, images[j].size(), CV_32FC1, map1j, map2j);

			Mat rectI, rectJ;
			cv::remap(images[i], rectI, map1i, map2i, INTER_LINEAR);
			cv::remap(images[j], rectJ, map1j, map2j, INTER_LINEAR);

			Mat grayI, grayJ;
			cvtColor(rectI, grayI, COLOR_BGR2GRAY);
			cvtColor(rectJ, grayJ, COLOR_BGR2GRAY);

			const int blockSize = 5;
			const int p1 = 8 * 3 * blockSize * blockSize;
			const int p2 = 32 * 3 * blockSize * blockSize;

			for (const int numDisparities : disparityRanges)
			{
				// the disparity map is as wide as the images
				if (numDisparities + blockSize >= grayI.cols)
					continue;

				// the arguments after the smoothing penalties are disp12MaxDiff,
				// preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange and
				// the mode; a too large uniquenessRatio rejects every disparity
				auto sgbm = cv::StereoSGBM::create(0, numDisparities, blockSize, p1, p2,
					1, 31, 10, 100, 2, cv::StereoSGBM::MODE_SGBM_3WAY);

				Mat disparity;
				sgbm->compute(grayI, grayJ, disparity);

				// the leftmost columns have no counterpart in the right image
				// because the disparity is positive
				disparity.colRange(0, numDisparities).setTo(-1);

				// the disparity is a fixed point value with four fractional bits
				std::vector<int> histogram(numDisparities, 0);
				int nvalid = 0;
				for (int y = 0; y < disparity.rows; y++)
				{
					const short* dptr = disparity.ptr<short>(y);
					for (int x = 0; x < disparity.cols; x++)
					{
						const int d = dptr[x] >> 4;
						if (d <= 0 || d >= numDisparities)
							continue;
						histogram[d]++;
						nvalid++;
					}
				}
				if (nvalid < 100)
					continue;

				// the dominant disparity is the peak of the smoothed histogram;
				// a tilted scene spreads the disparities over several bins
				std::vector<double> smoothed(numDisparities, 0.0);
				for (int d = 0; d < numDisparities; d++)
				{
					int sum = 0;
					int count = 0;
					for (int k = -2; k <= 2; k++)
					{
						const int dd = d + k;
						if (dd < 0 || dd >= numDisparities)
							continue;
						sum += histogram[dd];
						count++;
					}
					smoothed[d] = static_cast<double>(sum) / count;
				}

				int mode = 0;
				for (int d = 1; d < numDisparities; d++)
				{
					if (smoothed[d] > smoothed[mode])
						mode = d;
				}

				// sub-bin accuracy: the mean disparity around the peak
				double weighted = 0.0;
				double weight = 0.0;
				for (int d = std::max(0, mode - 3); d <= std::min(numDisparities - 1, mode + 3); d++)
				{
					weighted += static_cast<double>(d) * histogram[d];
					weight += histogram[d];
				}
				if (weight < 1e-6)
					continue;

				const double modalDisparity = weighted / weight;
				const double score = weight / (disparity.rows * disparity.cols);
				if (score > bestScore)
				{
					bestScore = score;
					bestMode = modalDisparity;
					bestBaseline = baseline;
					bestFocal = focal;
					bestPair0 = i;
					bestPair1 = j;
				}
			}
		}
	}

	if (bestPair0 < 0 || bestMode <= 0 || bestScore < 0.02)
	{
		result.detail = "the stereo matching did not find a reliable correspondence";
		return result;
	}

	const double depth = bestFocal * bestBaseline / bestMode;
	if (!(depth > 0.0) || !std::isfinite(depth))
	{
		result.detail = "the stereo matching returned an invalid depth";
		return result;
	}

	result.valid = true;
	result.minZ = depth * 0.7;
	result.maxZ = depth * 1.4;
	result.detail = format("camera %d and %d, disparity %.2f px, baseline %.4f, depth %.4f",
		bestPair0, bestPair1, bestMode, bestBaseline, depth);
	return result;
}

/** @brief Fallback depth range derived from the size of the camera rig.
*/
static DepthRange estimateDepthRangeFromRig(const std::vector<Vec3d>& centers)
{
	DepthRange result;

	Vec3d center(0.0, 0.0, 0.0);
	for (const auto& t : centers)
		center += t;
	center /= static_cast<double>(std::max<size_t>(centers.size(), 1));

	double radius = 0.0;
	for (const auto& t : centers)
		radius = std::max(radius, std::sqrt((t[0] - center[0]) * (t[0] - center[0]) +
			(t[1] - center[1]) * (t[1] - center[1]) +
			(t[2] - center[2]) * (t[2] - center[2])));
	if (radius < 1e-6)
		radius = 1.0;

	result.valid = true;
	result.minZ = 0.5 * radius;
	result.maxZ = 10.0 * radius;
	result.detail = format("estimated from the rig radius %.4f", radius);
	return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Per camera setup
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CameraSetup
{
	Mat K;            //!< intrinsic matrix of the processed images
	Mat map1, map2;   //!< undistortion maps
	bool useMaps = false;
	Size outputSize;  //!< size of the processed images
};

/** @brief Builds the remap tables of the equidistant (fisheye) lens model.
OpenCV 4.10 does not ship the public fisheye header any more, so the model is
evaluated here: the distorted radius is
    r_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
with theta = atan(r) the angle of the ray.
@param newK The camera matrix of the undistorted image.
@param distortion The distortion coefficients (k1, k2, k3, k4).
@param size The size of the undistorted image.
@param map1 Output map of the x coordinates.
@param map2 Output map of the y coordinates.
*/
static void initFisheyeUndistortMap(const Mat& newK, const Mat& distortion, const Size& size, Mat& map1, Mat& map2)
{
	const double fx = newK.at<double>(0, 0);
	const double fy = newK.at<double>(1, 1);
	const double cx = newK.at<double>(0, 2);
	const double cy = newK.at<double>(1, 2);

	const int ncoeffs = static_cast<int>(distortion.rows * distortion.cols);
	const double k1 = (ncoeffs > 0) ? distortion.at<double>(0) : 0.0;
	const double k2 = (ncoeffs > 1) ? distortion.at<double>(1) : 0.0;
	const double k3 = (ncoeffs > 2) ? distortion.at<double>(2) : 0.0;
	const double k4 = (ncoeffs > 3) ? distortion.at<double>(3) : 0.0;

	map1.create(size, CV_32FC1);
	map2.create(size, CV_32FC1);

	for (int v = 0; v < size.height; v++)
	{
		for (int u = 0; u < size.width; u++)
		{
			const double x = (u - cx) / fx;
			const double y = (v - cy) / fy;
			const double r = std::sqrt(x * x + y * y);

			if (r < 1e-12)
			{
				map1.at<float>(v, u) = static_cast<float>(u);
				map2.at<float>(v, u) = static_cast<float>(v);
				continue;
			}

			const double theta = std::atan(r);
			const double theta2 = theta * theta;
			const double rd = theta * (1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))));
			const double scale = rd / r;

			map1.at<float>(v, u) = static_cast<float>(fx * scale * x + cx);
			map2.at<float>(v, u) = static_cast<float>(fy * scale * y + cy);
		}
	}
}

static CameraSetup setupCamera(const Mat& calibration, const Mat& distortion, DistortionModel model,
	const Size& videoSize, int maxImageSize, bool undistort)
{
	CameraSetup setup;

	double scale = 1.0;
	if (maxImageSize > 0)
	{
		const int size = std::max(videoSize.width, videoSize.height);
		if (size > maxImageSize)
			scale = 1.0 * maxImageSize / size;
	}

	Size outputSize(cvRound(videoSize.width * scale), cvRound(videoSize.height * scale));
	// the propagation of the PatchMatch algorithm is a checkerboard, so the
	// width of the processed images has to be even
	outputSize.width &= ~1;
	checkTrue(outputSize.width >= 2 && outputSize.height >= 2, "The processed images are too small");
	setup.outputSize = outputSize;

	if (!undistort)
	{
		setup.K = calibration.clone();
		setup.K(Range(0, 2), Range(0, 3)) *= scale;
		setup.useMaps = false;
		return setup;
	}

	Mat newK;
	if (model == DISTORTION_FISHEYE)
	{
		// the equidistant model keeps the original camera matrix
		newK = calibration.clone();
	}
	else
	{
		newK = cv::getOptimalNewCameraMatrix(calibration, distortion, videoSize, 0.0);
	}

	newK.convertTo(newK, CV_64F);
	newK(Range(0, 2), Range(0, 3)) *= scale;

	if (model == DISTORTION_FISHEYE)
	{
		initFisheyeUndistortMap(newK, distortion, outputSize, setup.map1, setup.map2);
	}
	else
	{
		cv::initUndistortRectifyMap(calibration, distortion, Mat(), newK, outputSize, CV_32FC1,
			setup.map1, setup.map2);
	}

	setup.K = newK;
	setup.useMaps = true;
	return setup;
}

static Mat processFrame(const Mat& frame, const CameraSetup& setup)
{
	Mat dst;
	if (setup.useMaps)
	{
		cv::remap(frame, dst, setup.map1, setup.map2, INTER_LINEAR);
	}
	else
	{
		cv::resize(frame, dst, setup.outputSize, 0, 0,
			setup.outputSize == frame.size() ? INTER_NEAREST : INTER_AREA);
	}
	return dst;
}

/** @brief Parses an axis argument such as "x", "-y" or "+z".
The returned vector is the direction in the coordinate frame the string refers
to, i.e. one of the six signed coordinate axes.
@param axis The axis argument.
*/
static Vec3d parseAxis(const std::string& axis)
{
	std::string name = axis;
	double sign = 1.0;
	if (!name.empty() && (name[0] == '+' || name[0] == '-'))
	{
		sign = (name[0] == '-') ? -1.0 : 1.0;
		name = name.substr(1);
	}
	std::transform(name.begin(), name.end(), name.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });

	if (name == "x")
		return Vec3d(sign, 0.0, 0.0);
	if (name == "y")
		return Vec3d(0.0, sign, 0.0);
	if (name == "z")
		return Vec3d(0.0, 0.0, sign);

	CV_Error(Error::StsBadArg, format("The axis must be one of x, y, z, -x, -y or -z but is '%s'",
		axis.c_str()).c_str());
	return Vec3d();
}

/** @brief Rotates a point cloud into the output coordinate frame.
The transformation is a rigid rotation about the origin of the frame, so the
geometry of the cloud (all distances and angles between points) is unchanged;
only the coordinates are expressed in the new frame.
@param points The cloud; in place, any Mat with three channels and one element per point.
@param rotation Columns are the x, y and z axis of the output frame, expressed in the
input (world) frame.
@param origin The origin of the output frame in the input (world) frame.
*/
static void transformPoints(Mat& points, const Matx33d& rotation, const Vec3d& origin)
{
	if (points.empty())
		return;
	CV_Assert(points.channels() == 3 && points.isContinuous());

	// the cloud is a 1xN or Nx1 Mat with three channels
	const int npoints = static_cast<int>(points.total());
	Mat flat = points.reshape(3, npoints);
	const Matx33d rotationT = rotation.t();
	for (int i = 0; i < npoints; i++)
	{
		const Vec3f p = flat.at<Vec3f>(i);
		const Vec3d q = rotationT * Vec3d(p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]);
		flat.at<Vec3f>(i) = Vec3f(static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]));
	}
}

} // namespace cv

static void run(int argc, char* argv[])
{
	using namespace cv;

	const CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		return;
	}

	// get parameters
	const auto inputDir = parser.get<std::string>("@input-directory");
	const auto calibrationFile = parser.get<std::string>("calibration-file");
	const auto outputDirOption = parser.get<std::string>("output-directory");
	const auto method = parser.get<int>("method");
	const auto pmIters = parser.get<int>("pm-iterations");
	const auto gcIters = parser.get<int>("gc-iterations");
	const auto multiScale = parser.get<int>("multi-scale") == 1;
	const auto maxImageSize = parser.get<int>("max-image-size");
	const auto maxNeighbors = parser.get<int>("max-neighbors");
	const auto minDepth = parser.get<double>("min-depth");
	const auto maxDepth = parser.get<double>("max-depth");
	const auto frameStart = parser.get<int>("frame-start");
	const auto frameEnd = parser.get<int>("frame-end");
	const auto frameStep = parser.get<int>("frame-step");
	const auto undistort = parser.get<int>("undistort") == 1;
	const auto distortionModelOption = parser.get<std::string>("distortion-model");
	const auto poseConvention = parser.get<std::string>("pose-convention");
	const auto savePly = parser.get<int>("save-ply") == 1;
	const auto saveDepthVideo = parser.get<int>("save-depth-video") == 1;
	const auto rigPlot = parser.get<int>("rig-plot") == 1;
	const auto rigPlotFile = parser.get<std::string>("rig-plot-file");
	const auto debugView = parser.has("debug");
	auto nthreads = parser.get<int>("num-threads");

	if (!parser.check())
	{
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	checkTrue(frameStep > 0, "The frame step must be positive");
	checkTrue(maxNeighbors >= 1, "At least one neighbor view is required");
	checkTrue(poseConvention == "camera-to-world" || poseConvention == "world-to-camera",
		"The pose convention must be camera-to-world or world-to-camera");

	// to keep the point cloud identical in shape and distances, the output
	// frame is a rotated copy of the world frame: the origin is the reference
	// camera centre, +z is the optical axis of the reference camera (its viewing
	// direction) and an arbitrary axis is aligned to the requested up direction
	// by a rotation about z
	const int refCam = parser.get<int>("reference-camera");
	const auto upAxisArg = parser.get<std::string>("up-axis");
	const auto upAxisFrame = parser.get<std::string>("up-axis-frame");
	const Vec3d upRef = parseAxis(upAxisArg);
	checkTrue(upAxisFrame == "camera" || upAxisFrame == "world",
		"The up axis frame must be camera or world");

	// the default of an unset option is the literal "<none>"
	const String outputDir = (outputDirOption.empty() || outputDirOption == "<none>") ? inputDir : outputDirOption;

	if (nthreads < 0)
		nthreads = getNumThreads();
	omp_set_num_threads(nthreads);

	// a relative calibration file is looked up in the current directory first
	// and then in the input directory
	String calibrationPath = calibrationFile;
	if (!std::filesystem::path(calibrationPath).is_absolute() &&
		!std::filesystem::exists(calibrationPath))
	{
		const String candidate = (std::filesystem::path(inputDir) / calibrationPath).string();
		if (std::filesystem::exists(candidate))
			calibrationPath = candidate;
	}

	const char* methodStr[] = { "GIPUMA_FAST", "ACMH" };

	std::cout << "==================== configulations ====================" << std::endl;
	std::cout << "input directory                  : " << inputDir << std::endl;
	std::cout << "calibration file                 : " << calibrationPath << std::endl;
	std::cout << "output directory                 : " << outputDir << std::endl;
	std::cout << "PatchMatch method                : " << methodStr[method] << std::endl;
	std::cout << "PatchMatch iterations            : " << pmIters << std::endl;
	std::cout << "geometric consistency iterations : " << gcIters << std::endl;
	std::cout << "multi scale                      : " << (multiScale ? "Yes" : "No") << std::endl;
	std::cout << "maximum image size               : " << maxImageSize << std::endl;
	std::cout << "maximum number of neighbors       : " << maxNeighbors << std::endl;
	std::cout << "reference camera                 : " << refCam << std::endl;
	std::cout << "up axis                          : " << upAxisArg << " (" << upAxisFrame << " frame)"
		<< std::endl;
	std::cout << "undistort                        : " << (undistort ? "Yes" : "No") << std::endl;
	std::cout << "pose convention                  : " << poseConvention << std::endl;
	std::cout << "save point clouds                : " << (savePly ? "Yes" : "No") << std::endl;
	std::cout << "save depth map videos            : " << (saveDepthVideo ? "Yes" : "No") << std::endl;
	std::cout << "draw the camera rig layout       : " << (rigPlot ? "Yes" : "No") << std::endl;
	std::cout << "number of threads                : " << nthreads << std::endl;
	std::cout << "========================================================" << std::endl << std::endl;

	/////////////////////////////////////////////////////////////////////////////
	// load the calibration
	/////////////////////////////////////////////////////////////////////////////

	CameraCalibration calibration;
	const int ncameras = loadCameraCalibration(calibrationPath, calibration);
	checkTrue(refCam >= 0 && refCam < ncameras, "The reference camera is out of range");
	checkTrue(ncameras >= 2, "At least two cameras are required for multi-view stereo");

	std::vector<Mat> rotations;
	std::vector<Mat> translations;
	std::vector<Vec3d> centers;
	extractPoses(calibration.poses, poseConvention == "world-to-camera", rotations, translations, centers);

	std::cout << "number of cameras                : " << ncameras << std::endl;
	for (int i = 0; i < ncameras; i++)
	{
		std::cout << format("  camera_%d: centre (%.4f, %.4f, %.4f)\n", i,
			centers[i][0], centers[i][1], centers[i][2]).c_str();
	}
	std::cout << std::endl;

	/////////////////////////////////////////////////////////////////////////////
	// the neighbours of a camera are the cameras with the shortest baseline
	/////////////////////////////////////////////////////////////////////////////

	std::vector<std::vector<int>> neighborSets(ncameras);
	for (int i = 0; i < ncameras; i++)
	{
		std::vector<std::pair<double, int>> order;
		for (int j = 0; j < ncameras; j++)
		{
			if (i == j)
				continue;
			const double distance = std::sqrt(
				(centers[i][0] - centers[j][0]) * (centers[i][0] - centers[j][0]) +
				(centers[i][1] - centers[j][1]) * (centers[i][1] - centers[j][1]) +
				(centers[i][2] - centers[j][2]) * (centers[i][2] - centers[j][2]));
			order.emplace_back(distance, j);
		}
		std::sort(order.begin(), order.end());

		for (int k = 0; k < std::min<int>(maxNeighbors, static_cast<int>(order.size())); k++)
			neighborSets[i].push_back(order[k].second);
	}

	/////////////////////////////////////////////////////////////////////////////
	// open the videos
	/////////////////////////////////////////////////////////////////////////////

	std::vector<String> videoPaths(ncameras);
	std::vector<VideoCapture> captures(ncameras);

	for (int i = 0; i < ncameras; i++)
	{
		videoPaths[i] = findVideoFile(inputDir, i);
		checkTrue(!videoPaths[i].empty(),
			format("No video file found for camera_%d in '%s'", i, inputDir.c_str()));

		if (!captures[i].open(videoPaths[i]))
			CV_Error(Error::StsBadArg, format("Cannot open the video file '%s'", videoPaths[i].c_str()).c_str());

		std::cout << "  camera_" << i << " : " << videoPaths[i] << std::endl;
	}
	std::cout << std::endl;

	/////////////////////////////////////////////////////////////////////////////
	// determine the frame count and the image size
	/////////////////////////////////////////////////////////////////////////////

	int frameCount = -1;
	double fps = 30.0;
	Size videoSize;

	for (int i = 0; i < ncameras; i++)
	{
		const int count = static_cast<int>(captures[i].get(CAP_PROP_FRAME_COUNT));
		const double cameraFps = captures[i].get(CAP_PROP_FPS);
		if (count > 0)
			frameCount = (frameCount < 0) ? count : std::min(frameCount, count);
		if (cameraFps > 0.0)
			fps = (i == 0) ? cameraFps : std::min(fps, cameraFps);
	}

	// read the first frame of every camera to get the image size
	std::vector<Mat> firstFrames(ncameras);
	{
		for (int i = 0; i < ncameras; i++)
		{
			CV_Assert(captures[i].read(firstFrames[i]));
			checkTrue(!firstFrames[i].empty(),
				format("Cannot read a frame from '%s'", videoPaths[i].c_str()));
		}
		videoSize = firstFrames[0].size();
		for (int i = 1; i < ncameras; i++)
		{
			checkTrue(firstFrames[i].size() == videoSize,
				"All the videos must have the same resolution");
		}
	}

	if (frameCount < 0)
	{
		CV_Error(Error::StsBadArg, "Cannot determine the number of frames of the videos");
	}

	int lastFrame = frameCount - 1;
	if (frameEnd >= 0)
		lastFrame = std::min(lastFrame, frameEnd);
	lastFrame = std::min(lastFrame, frameStart + (lastFrame - frameStart) / frameStep * frameStep);

	std::cout << "video resolution                 : " << videoSize.width << " x " << videoSize.height << std::endl;
	std::cout << "number of frames                 : " << frameCount << std::endl;
	std::cout << "frames to process                : " << frameStart << " .. " << lastFrame << std::endl;
	std::cout << "frame rate                       : " << fps << std::endl;
	std::cout << std::endl;

	/////////////////////////////////////////////////////////////////////////////
	// set up the per camera undistortion
	/////////////////////////////////////////////////////////////////////////////

	std::vector<CameraSetup> setups(ncameras);
	std::vector<Mat> cameras(ncameras);

	for (int i = 0; i < ncameras; i++)
	{
		const DistortionModel model = resolveDistortionModel(calibration.distortionTypes[i],
			static_cast<int>(calibration.distortions[i].rows * calibration.distortions[i].cols),
			distortionModelOption);

		if (!calibration.imageSizes[i].empty() &&
			(calibration.imageSizes[i].width != videoSize.width ||
				calibration.imageSizes[i].height != videoSize.height))
		{
			std::cout << "warning: the calibration of camera_" << i << " was made for "
				<< calibration.imageSizes[i].width << " x " << calibration.imageSizes[i].height
				<< " but the video is " << videoSize.width << " x " << videoSize.height << std::endl;
		}

		setups[i] = setupCamera(calibration.cameras[i], calibration.distortions[i], model,
			videoSize, maxImageSize, undistort);
		cameras[i] = setups[i].K;

		std::cout << "  camera_" << i << " : distortion " << distortionModelName(model)
			<< ", processed at " << setups[i].outputSize.width << " x " << setups[i].outputSize.height
			<< std::endl;
	}
	std::cout << std::endl;

	/////////////////////////////////////////////////////////////////////////////
	// view id sets: every camera is a reference view, the neighbors are the
	// cameras with the shortest baseline
	/////////////////////////////////////////////////////////////////////////////

	std::vector<Mat> viewIdSets;
	for (int i = 0; i < ncameras; i++)
	{
		std::vector<int> viewIds;
		viewIds.push_back(i);
		for (const int j : neighborSets[i])
			viewIds.push_back(j);

		// Mat(vector) wraps the buffer without copying, so a copy is required
		// because viewIds is recreated in every iteration
		viewIdSets.push_back(Mat(viewIds, true));
	}

	/////////////////////////////////////////////////////////////////////////////
	// drawing of the camera rig: the distances and the angles between the
	// neighbouring cameras, so they can be measured in the real world
	/////////////////////////////////////////////////////////////////////////////

	if (rigPlot)
	{
		std::vector<RigCamera> rigCameras(ncameras);
		for (int i = 0; i < ncameras; i++)
		{
			rigCameras[i].center = centers[i];
			rigCameras[i].axis = opticalAxis(rotations[i]);
			const double fx = calibration.cameras[i].at<double>(0, 0);
			rigCameras[i].horizontalFov = (fx > 0.0)
				? 2.0 * std::atan(videoSize.width / (2.0 * fx)) * 180.0 / CV_PI
				: 0.0;
		}

		// the pairs of neighbouring cameras, every pair is listed once
		std::vector<std::pair<int, int>> pairs;
		for (int i = 0; i < ncameras; i++)
		{
			for (const int j : neighborSets[i])
			{
				const std::pair<int, int> pair(std::min(i, j), std::max(i, j));
				if (std::find(pairs.begin(), pairs.end(), pair) == pairs.end())
					pairs.push_back(pair);
			}
		}
		std::sort(pairs.begin(), pairs.end());

		const auto measurements = measureRig(rigCameras, pairs);
		printRigMeasurements(rigCameras, measurements);

		Mat rigImage;
		const String subtitle = format("%s, %s convention, %d cameras, %d neighbours per camera",
			calibrationPath.c_str(), poseConvention.c_str(), ncameras, maxNeighbors);
		drawRigLayout(rigCameras, pairs, measurements, "Camera rig layout", subtitle, rigImage);

		// a relative file name is written into the output directory
		String rigPath = rigPlotFile.empty() ? String("rig_layout.png") : rigPlotFile;
		if (!std::filesystem::path(rigPath).is_absolute())
		{
			makeDirectory(outputDir);
			rigPath = outputDir + "/" + rigPath;
		}
		checkTrue(imwrite(rigPath, rigImage), format("Cannot write the rig layout drawing '%s'", rigPath.c_str()).c_str());
		std::cout << "camera rig layout                : " << rigPath << std::endl << std::endl;
	}

	/////////////////////////////////////////////////////////////////////////////
	// depth range
	/////////////////////////////////////////////////////////////////////////////

	double searchMinZ = minDepth;
	double searchMaxZ = maxDepth;

	if (searchMinZ > 0.0 && searchMaxZ > searchMinZ)
	{
		std::cout << "depth range                      : [" << searchMinZ << ", " << searchMaxZ
			<< "] (given)" << std::endl << std::endl;
	}
	else
	{
		// the depth range is estimated from the feature correspondences of the
		// first frame; if they are not reliable (a textureless scene) the
		// rectified stereo match is used as a fallback
		std::vector<Mat> firstImages(ncameras);
		for (int i = 0; i < ncameras; i++)
			firstImages[i] = processFrame(firstFrames[i], setups[i]);

		DepthRange range = estimateDepthRangeFromFeatures(firstImages, cameras, rotations, centers);
		if (!range.valid)
		{
			std::cout << "warning: " << range.detail << ", trying a stereo match" << std::endl;
			range = estimateDepthRange(firstImages, cameras, rotations, centers);
		}
		if (!range.valid)
		{
			std::cout << "warning: " << range.detail << ", falling back to the rig size" << std::endl;
			range = estimateDepthRangeFromRig(centers);
		}
		searchMinZ = range.minZ;
		searchMaxZ = range.maxZ;
		std::cout << "depth range                      : [" << searchMinZ << ", " << searchMaxZ
			<< "] (" << range.detail << ")" << std::endl << std::endl;
	}

	/////////////////////////////////////////////////////////////////////////////
	// output coordinate frame
	/////////////////////////////////////////////////////////////////////////////

	// the output frame is a rotated copy of the calibration world frame, so the
	// geometry of the point cloud (distances and angles between points) is
	// unchanged.  Its origin is the centre of the reference camera, +z is the
	// optical axis of that camera (the viewing direction) and the requested axis
	// is used as up; +x follows from the right-handed frame.
	const Matx33d Rref = toMatx33(rotations[refCam]);
	const Vec3d outputOrigin = centers[refCam];
	const Vec3d zOut = normalize(Rref * Vec3d(0.0, 0.0, 1.0));

	Vec3d upWorld = (upAxisFrame == "camera") ? Rref * upRef : upRef;
	upWorld -= upWorld.dot(zOut) * zOut;
	checkTrue(cv::norm(upWorld) > 1e-6,
		"The up axis must not be the viewing direction of the reference camera");
	const Vec3d yOut = normalize(upWorld);
	const Vec3d xOut = normalize(yOut.cross(zOut));

	Matx33d outputRotation;
	outputRotation(0, 0) = xOut[0]; outputRotation(0, 1) = yOut[0]; outputRotation(0, 2) = zOut[0];
	outputRotation(1, 0) = xOut[1]; outputRotation(1, 1) = yOut[1]; outputRotation(1, 2) = zOut[1];
	outputRotation(2, 0) = xOut[2]; outputRotation(2, 1) = yOut[2]; outputRotation(2, 2) = zOut[2];

	std::cout << format("output frame                     : camera_%d, origin (%.4f, %.4f, %.4f), "
		"viewing +z (%.4f, %.4f, %.4f), up %s/%s (%.4f, %.4f, %.4f)\n", refCam,
		outputOrigin[0], outputOrigin[1], outputOrigin[2],
		zOut[0], zOut[1], zOut[2],
		upAxisFrame.c_str(), upAxisArg.c_str(), yOut[0], yOut[1], yOut[2]) << std::endl;

	/////////////////////////////////////////////////////////////////////////////
	// run PatchMatch
	/////////////////////////////////////////////////////////////////////////////

	auto mvs = cuda::PatchMatchMVS::create();
	mvs->setPatchMatchMethod(static_cast<cv::cuda::PatchMatchMethod>(method));
	mvs->setPatchMatchIters(pmIters);
	mvs->setGeometricConsistencyIters(gcIters);
	mvs->setMultiScale(multiScale);
	mvs->setDepthRange(searchMinZ, searchMaxZ);
	if (debugView)
		mvs->setDebugCallBack(debugCallBack);

	/////////////////////////////////////////////////////////////////////////////
	// output
	/////////////////////////////////////////////////////////////////////////////

	const String plyDir = outputDir + "/ply";
	const String depthDir = outputDir + "/depth";
	if (savePly)
		makeDirectory(plyDir);
	if (saveDepthVideo)
		makeDirectory(depthDir);

	/////////////////////////////////////////////////////////////////////////////
	// frame loop
	/////////////////////////////////////////////////////////////////////////////

	// the captures are positioned after the first frame, which was already read
	std::vector<int> currentFrames(ncameras, 1);
	std::vector<Mat> images(ncameras);
	std::vector<VideoWriter> depthWriters(ncameras);

	const int nframes = (lastFrame >= frameStart) ? ((lastFrame - frameStart) / frameStep + 1) : 0;

	for (int f = 0; f < nframes; f++)
	{
		const int frameIndex = frameStart + f * frameStep;

		// read the requested frame of every camera (written per thread)
		std::vector<uchar> frameOk(ncameras, 1);
#pragma omp parallel for
		for (int i = 0; i < ncameras; i++)
		{
			if (frameIndex == 0)
			{
				// the first frame has already been read
				images[i] = processFrame(firstFrames[i], setups[i]);
				continue;
			}

			Mat frame;
			// advance the capture to the requested frame
			while (currentFrames[i] < frameIndex)
			{
				if (!captures[i].read(frame))
				{
					frameOk[i] = 0;
					break;
				}
				currentFrames[i]++;
			}
			if (!frameOk[i])
				continue;
			if (!captures[i].read(frame))
			{
				frameOk[i] = 0;
				continue;
			}
			currentFrames[i]++;
			images[i] = processFrame(frame, setups[i]);
		}

		if (std::count(frameOk.begin(), frameOk.end(), 0) > 0)
		{
			std::cout << "warning: cannot read frame " << frameIndex << ", stopping" << std::endl;
			break;
		}

		Mat densePoints, denseColors, denseNormals;
		std::vector<Mat> depths;
		mvs->compute(images, cameras, rotations, translations, viewIdSets, Mat(),
			densePoints, denseColors, denseNormals, depths);

		std::cout << "frame " << frameIndex << " : " << densePoints.total() << " points" << std::endl;

		// express the cloud in the requested output frame; the geometry itself
		// is untouched because this is a rigid transformation
		if (!densePoints.empty())
		{
			transformPoints(densePoints, outputRotation, outputOrigin);
			if (!denseNormals.empty() && denseNormals.channels() == 3 && denseNormals.isContinuous())
			{
				// normals are rotated but not translated
				const int nnormals = static_cast<int>(denseNormals.total());
				Mat flat = denseNormals.reshape(3, nnormals);
				const Matx33d rotationT = outputRotation.t();
				for (int i = 0; i < nnormals; i++)
				{
					const Vec3f v = flat.at<Vec3f>(i);
					const Vec3d w = rotationT * Vec3d(v[0], v[1], v[2]);
					flat.at<Vec3f>(i) = Vec3f(static_cast<float>(w[0]), static_cast<float>(w[1]), static_cast<float>(w[2]));
				}
			}
		}

		// dense point cloud
		if (savePly && !densePoints.empty())
		{
			const String path = format("%s/frame_%06d.ply", plyDir.c_str(), frameIndex);
			viz::writeCloud(path, densePoints, denseColors);
		}

		// depth map videos
		if (saveDepthVideo)
		{
			for (int i = 0; i < ncameras; i++)
			{
				if (i >= static_cast<int>(depths.size()) || depths[i].empty())
					continue;

				const Mat color = colorizeDepth(depths[i], searchMinZ, searchMaxZ);
				if (color.empty())
					continue;

				if (!depthWriters[i].isOpened())
				{
					const String path = format("%s/camera_%d.mp4", depthDir.c_str(), i);
					if (openVideoWriter(path, fps, color.size(), depthWriters[i]) != 0)
					{
						std::cout << "warning: cannot open the depth map video writer for camera_" << i << std::endl;
						continue;
					}
				}

				depthWriters[i].write(color);
			}
		}
	}

	for (auto& writer : depthWriters)
	{
		if (writer.isOpened())
			writer.release();
	}

	std::cout << std::endl << "done" << std::endl;
}

int main(int argc, char* argv[])
{
	try
	{
		run(argc, argv);
	}
	catch (const cv::Exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

