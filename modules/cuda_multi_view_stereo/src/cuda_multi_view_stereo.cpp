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

#include "cuda_multi_view_stereo.h"

#include <set>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#include "debug_print.h"
#include "optimization_config.h"
#include "propagation_gipuma.h"
#include "propagation_acmh.h"
#include "texture_image.h"

#include "cuda_multi_view_stereo_internal.h"

#define DEBUG_CALLBACK(I, Z, N, C, TITLE) if (callBack_) callBack_(I, Z, N, C, TITLE);

namespace cv
{
namespace cuda
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr float EPS = 1e-10f;
static constexpr double LAMBDA_MIN = 0.75;
static constexpr double LAMBDA_MAX = 1.25;
static constexpr double C_MIN = 0.01;
static constexpr double C_MAX = 0.99;
static constexpr float DELTA_DEPTH = 1e-5f;
static constexpr float PI_F = static_cast<float>(CV_PI);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Static functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static inline float floatCast(T v)
{
	return static_cast<float>(v);
}

static void getInputMatVectors(InputArrayOfArrays _images, InputArrayOfArrays _cameras, InputArrayOfArrays _rotations, InputArrayOfArrays _translations, InputArrayOfArrays _viewIdSets,
	std::vector<Mat>& images, std::vector<Mat>& cameras, std::vector<Mat>& rotations, std::vector<Mat>& translations, std::vector<Mat>& viewIdSets)
{
	// convert input arrays to mat vector
	_images.getMatVector(images);
	_cameras.getMatVector(cameras);
	_rotations.getMatVector(rotations);
	_translations.getMatVector(translations);
	_viewIdSets.getMatVector(viewIdSets);

	// needs at least 2 images
	const int nimages = static_cast<int>(images.size());
	CV_Assert(nimages >= 2);

	// needs at least 1 reference view
	const int nreferences = static_cast<int>(viewIdSets.size());
	CV_Assert(nreferences >= 1);

	// number of images, cameras, rotations, and translations must be the same
	CV_Assert(cameras.size() == images.size());
	CV_Assert(rotations.size() == images.size());
	CV_Assert(translations.size() == images.size());

	// check matrix size and type
	for (int i = 0; i < nimages; i++)
	{
		const auto& I = images[i];
		auto& K = cameras[i];
		auto& R = rotations[i];
		auto& t = translations[i];

		K = K.reshape(1, 3);
		R = R.reshape(1, 3);
		t = t.reshape(1, 3);

		CV_Assert(I.type() == CV_8UC3);
		CV_Assert(K.size() == Size(3, 3) && (K.type() == CV_32F || K.type() == CV_64F));
		CV_Assert(R.size() == Size(3, 3) && (R.type() == CV_32F || R.type() == CV_64F));
		CV_Assert(t.size() == Size(1, 3) && (t.type() == CV_32F || t.type() == CV_64F));
	}

	for (int i = 0; i < nreferences; i++)
	{
		const auto& viewIds = viewIdSets[i];
		CV_Assert(viewIds.type() == CV_32S);
		CV_Assert(viewIds.checkVector(1) >= 2);
	}
}

static std::set<int> getActiveViewIndices(const std::vector<Mat>& connections)
{
	std::set<int> indexSet;
	for (const auto& viewIndices : connections)
	{
		const int nviews = viewIndices.checkVector(1);
		for (int i = 0; i < nviews; i++)
			indexSet.insert(viewIndices.at<int>(i));
	}
	return indexSet;
}

static int calcPyramidLevels(const std::vector<Mat>& images, std::vector<int>& levels, int sizeBound = 500)
{
	const int nimages = static_cast<int>(images.size());
	levels.resize(nimages);

	auto calcPyramidLevel = [](int imgSize, int sizeBound)
	{
		int level = 0;

		for (int size = imgSize; size > sizeBound; size >>= 1)
			level++;

		return std::max(level - 1, 0);
	};

	int maxLevel = 0;
	for (int i = 0; i < nimages; i++)
	{
		const int level = calcPyramidLevel(std::max(images[i].cols, images[i].rows), sizeBound);
		maxLevel = std::max(maxLevel, level);
		levels[i] = level;
	}
	return maxLevel;
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

static void convertToGray(const Mat& src, Mat& dst)
{
	switch (src.type())
	{
	case CV_8UC1:
		dst = src;
		break;
	case CV_8UC3:
		cvtColor(src, dst, COLOR_BGR2GRAY);
		break;
	case CV_8UC4:
		cvtColor(src, dst, COLOR_BGRA2GRAY);
		break;
	default:
		CV_Error(Error::StsBadArg, "Image should be 8UC1, 8UC3 or 8UC4");
	}
}

void featureTransform(const Mat& src, Mat& dst)
{
	convertToGray(src, dst);
}

static void projectToCamera(const Mat& _points, const Matx33d& _R, const Vec3d& _t, std::vector<Vec3f>& Xcs)
{
	CV_Assert(_points.type() == CV_32FC3 || _points.type() == CV_64FC3);

	const auto R = Matx33f(_R.t());
	const auto t = Vec3f(-_R.t() * _t);

	Mat points;
	if (_points.type() == CV_32FC3)
		points = _points;
	else
		_points.convertTo(points, CV_32FC3);

	const int npoints = points.checkVector(3);
	const Vec3f* ptrPoints = points.ptr<Vec3f>();

	Xcs.clear();
	Xcs.reserve(npoints);

	for (int i = 0; i < npoints; i++)
	{
		const auto Xw = ptrPoints[i];
		const auto Xc = R * Xw + t;
		if (Xc(2) > 0)
			Xcs.push_back(Xc);
	}
}

static inline float sqnorm(float x, float y)
{
	return x * x + y * y;
}

static void fuseDepthMaps(const std::vector<Mat>& images, const std::vector<Mat>& cameras, const std::vector<Mat>& rotations,
	const std::vector<Mat>& translations, const Mat& viewIndices, const std::vector<Mat>& depths, const std::vector<Mat>& normals,
	std::vector<Mat>& masks, std::vector<Vec3f>& points3D, std::vector<Vec3b>& colors3D, std::vector<Vec3f>& normals3D,
	float maxDepthError = 0.01f, float maxReprojectError = 2.f, float maxNormalAngleDiff = 10.f)
{
	maxReprojectError *= maxReprojectError;

	const float minNormalCos = cosf((PI_F / 180) * maxNormalAngleDiff);

	// get views
	std::vector<Mat> Is;
	std::vector<Matx33f> Ks;
	std::vector<Matx33f> Rs;
	std::vector<Vec3f> ts;
	std::vector<Mat> Ds;
	std::vector<Mat> Ns;
	std::vector<Mat> Ms;

	int nviews = 0;
	for (int i = 0; i < viewIndices.checkVector(1); i++)
	{
		const int viewIndex = viewIndices.at<int>(i);
		if (!masks[viewIndex].empty())
		{
			Is.push_back(images[viewIndex]);
			Ks.push_back(cameras[viewIndex]);
			Rs.push_back(rotations[viewIndex]);
			ts.push_back(translations[viewIndex]);
			Ds.push_back(depths[viewIndex]);
			Ns.push_back(normals[viewIndex]);
			Ms.push_back(masks[viewIndex]);
			nviews++;
		}
	}
	CV_Assert(nviews >= 2);

	const int minCount = std::min(nviews - 1, 2);

	std::vector<Matx33f> Rwc(nviews), Rcw(nviews);
	std::vector<Vec3f> twc(nviews), tcw(nviews);
	for (int i = 0; i < nviews; i++)
	{
		Rwc[i] = Rs[i] * Ks[i].inv();
		twc[i] = ts[i];
		Rcw[i] = Ks[i] * Rs[i].t();
		tcw[i] = -Ks[i] * Rs[i].t() * ts[i];
	}

	// get reference view parameters
	const auto& I1 = Is[0];
	const auto& D1 = Ds[0];
	const auto& N1 = Ns[0];
	auto& M1 = Ms[0];

#pragma omp parallel for
	for (int y1 = 0; y1 < I1.rows; y1++)
	{
		const Vec3b* ptrI1 = I1.ptr<Vec3b>(y1);
		const float* ptrD1 = D1.ptr<float>(y1);
		const Vec3f* ptrN1 = N1.ptr<Vec3f>(y1);
		uchar* ptrM1 = M1.ptr<uchar>(y1);

		std::vector<Point> matchedPts(nviews);

		for (int x1 = 0; x1 < I1.cols; x1++)
		{
			if (!ptrM1[x1])
				continue;

			// reproject reference point to world coordinate
			const auto Z1 = ptrD1[x1];
			const auto n1 = ptrN1[x1];
			const auto Xw = Rwc[0] * Vec3f(Z1 * x1, Z1 * y1, Z1) + twc[0];
			const auto Nw = Rs[0] * n1;

			int count = 0;
			Vec3f sumX = Xw;
			Vec3f sumC = ptrI1[x1];
			Vec3f sumN = Nw;
			for (int i = 1; i < nviews; i++)
			{
				matchedPts[i] = Point(-1, -1);

				// get target view parameters
				const auto& I2 = Is[i];
				const auto& D2 = Ds[i];
				const auto& N2 = Ns[i];
				const auto& M2 = Ms[i];

				// project into neighbor camera coordinate
				const auto Xc2 = Rcw[i] * Xw + tcw[i];
				const auto Z2 = Xc2(2);
				if (Z2 <= 0)
					continue;

				const auto invZ2 = 1.f / Z2;
				const auto x2 = cvRound(invZ2 * Xc2(0));
				const auto y2 = cvRound(invZ2 * Xc2(1));

				// check if inside image
				if (!(x2 >= 0 && x2 < I2.cols && y2 >= 0 && y2 < I2.rows))
					continue;

				if (!M2.at<uchar>(y2, x2))
					continue;

				// again, reproject to reference camera coordinate
				const auto _Z2 = D2.at<float>(y2, x2);
				const auto _Xw = Rwc[i] * Vec3f(_Z2 * x2, _Z2 * y2, _Z2) + twc[i];
				const auto _Xc1 = Rcw[0] * _Xw + tcw[0];
				const auto _Z1 = _Xc1(2);

				if (_Z1 <= 0 || fabsf(_Z1 - Z1) > maxDepthError * Z1)
					continue;

				const auto _invZ1 = 1.f / _Z1;
				const auto _x1 = _invZ1 * _Xc1(0);
				const auto _y1 = _invZ1 * _Xc1(1);

				if (sqnorm(_x1 - x1, _y1 - y1) > maxReprojectError)
					continue;

				const auto n2 = N2.at<Vec3f>(y2, x2);
				const auto _Nw = Rs[i] * n2;

				if (Nw.dot(_Nw) < minNormalCos)
					continue;

				count++;
				sumX += _Xw;
				sumC += I2.at<Vec3b>(y2, x2);
				sumN += _Nw;
				matchedPts[i] = Point(x2, y2);
			}

			if (count >= minCount)
			{
				ptrM1[x1] = 0;

				for (int i = 1; i < nviews; i++)
				{
					const auto pt = matchedPts[i];
					if (pt.x >= 0)
						Ms[i].at<uchar>(pt) = 0;
				}

				const float invn = 1.f / (count + 1);
#pragma omp critical
				{
					points3D.push_back(invn * sumX);
					colors3D.push_back(invn * sumC);
					normals3D.push_back(normalize(invn * sumN));
				}
			}
		}
	}
}

void reprojectPoints(const std::vector<Mat>& images, const std::vector<Mat>& cameras,
	const std::vector<Mat>& rotations, const std::vector<Mat>& translations, const Mat& viewIndices,
	const std::vector<Mat>& depths, const std::vector<Mat>& normals, const std::vector<Mat>& masks,
	std::vector<Vec3f>& points3D, std::vector<Vec3b>& colors3D, std::vector<Vec3f>& normals3D)
{
	// get reference view parameters
	const int referenceIndex = viewIndices.at<int>(0);
	const auto& I1 = images[referenceIndex];
	const auto& D1 = depths[referenceIndex];
	const auto& N1 = normals[referenceIndex];
	const auto& M1 = masks[referenceIndex];

	const auto K1 = Matx33f(cameras[referenceIndex]);
	const auto R1 = Matx33f(rotations[referenceIndex]);

	const auto Rwc = R1 * K1.inv();
	const auto twc = Vec3f(translations[referenceIndex]);

	for (int y1 = 0; y1 < I1.rows; y1++)
	{
		const Vec3b* ptrI1 = I1.ptr<Vec3b>(y1);
		const float* ptrD1 = D1.ptr<float>(y1);
		const Vec3f* ptrN1 = N1.ptr<Vec3f>(y1);
		const uchar* ptrM1 = M1.ptr<uchar>(y1);

		for (int x1 = 0; x1 < I1.cols; x1++)
		{
			if (!ptrM1[x1])
				continue;

			// reproject reference point to world coordinate
			const auto Z1 = ptrD1[x1];
			const auto Xw = Rwc * Vec3f(Z1 * x1, Z1 * y1, Z1) + twc;
			const auto n1 = ptrN1[x1];
			const auto Nw = R1 * n1;

			points3D.push_back(Xw);
			colors3D.push_back(ptrI1[x1]);
			normals3D.push_back(Nw);
		}
	}
}

static void uploadImages(const std::vector<Mat>& images, std::vector<TextureImage>& d_images)
{
	const int nimages = static_cast<int>(images.size());

	d_images.resize(nimages);

	std::vector<gpu::TexObjSz> texs(nimages);
	for (int i = 0; i < nimages; i++)
	{
		const Mat& image = images[i];
		d_images[i].create(image, cudaAddressModeWrap, cudaFilterModeLinear, cudaReadModeNormalizedFloat);
		texs[i] = { d_images[i].getTextureObject(), image.cols, image.rows };
	}
	gpu::uploadImages(texs);
}

static void uploadDepths(const std::vector<Mat>& depths, std::vector<TextureImage>& d_depths)
{
	const int nimages = static_cast<int>(depths.size());

	d_depths.resize(nimages);

	std::vector<gpu::TexObjSz> texs(nimages);
	for (int i = 0; i < nimages; i++)
	{
		const Mat& image = depths[i];
		d_depths[i].create(image, cudaAddressModeBorder, cudaFilterModePoint, cudaReadModeElementType);
		texs[i] = { d_depths[i].getTextureObject(), image.cols, image.rows };
	}
	gpu::uploadDepths(texs);
}

static void uploadHomographyParams(const std::vector<Matx33d>& Ks, const std::vector<Matx33d>& Rs, const std::vector<Vec3d>& ts)
{
	const int nothers = static_cast<int>(Ks.size()) - 1;

	const auto& K1 = Ks[0];
	const auto& R1 = Rs[0];
	const auto& t1 = ts[0];

	// elements of K1^-1
	Vec4f invK1;
	invK1[0] = floatCast(1.f / K1(0, 0));
	invK1[1] = floatCast(1.f / K1(1, 1));
	invK1[2] = floatCast(-K1(0, 2) / K1(0, 0));
	invK1[3] = floatCast(-K1(1, 2) / K1(1, 1));

	// pre-compute projection matrix
	std::vector<Matx33f> R21(nothers), R12(nothers);
	std::vector<Vec3f> t21(nothers), t12(nothers);
	for (int i = 0; i < nothers; i++)
	{
		const auto& K2 = Ks[i + 1];
		const auto& R2 = Rs[i + 1];
		const auto& t2 = ts[i + 1];

		R21[i] = Matx33f(K2 * R2.t() * R1 * K1.inv());
		t21[i] = Vec3f(K2 * R2.t() * (t1 - t2));
		R12[i] = Matx33f(K1 * R1.t() * R2 * K2.inv());
		t12[i] = Vec3f(K1 * R1.t() * (t2 - t1));
	}

	gpu::uploadHomographyParams(invK1, R21, t21, R12, t12);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of PatchMatchMVS
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class  PatchMatchMVSImpl : public PatchMatchMVS
{
public:

	PatchMatchMVSImpl(PatchMatchMethod method, int pmIters, int gcIters, bool multiScale)
		: method_(method), pmIters_(pmIters), gcIters_(gcIters), multiScale_(multiScale)
	{
		callBack_ = nullptr;
		hasDepthRange_ = false;
		depthRangeMin_ = 0.f;
		depthRangeMax_ = 0.f;
	}

	Ptr<RedBlackPropagation> createPropagation(const std::vector<Mat>& Fs, const std::vector<Matx33d>& Ks,
		const std::vector<Matx33d>& Rs, const std::vector<Vec3d>& ts, const std::vector<Mat>& Ds, float minZ, float maxZ, bool geom)
	{
		if (method_ == GIPUMA_FAST)
			return PropagationGipuma::create(Fs, Ks, Rs, ts, minZ, maxZ, 6, geom);
		else if (method_ == ACMH)
			return PropagationACMH::create(Fs, Ks, Rs, ts, Ds, minZ, maxZ, geom);
		else
			CV_Error(Error::StsBadArg, "No such method");
	}

	void estimateDepth(const std::vector<Mat>& images, const std::vector<Mat>& cameras,
		const std::vector<Mat>& rotations, const std::vector<Mat>& translations, const std::vector<Mat>& features,
		int referenceIdx, const Mat& viewIndices, const Mat& sparsePoints,
		std::vector<Mat>& depthMaps, std::vector<Mat>& normalMaps, bool geom, bool randomInit)
	{
		const int nviews = viewIndices.checkVector(1);

		// get views
		std::vector<Mat> Is;
		std::vector<Matx33d> Ks;
		std::vector<Matx33d> Rs;
		std::vector<Vec3d> ts;
		std::vector<Mat> Ds;
		std::vector<Mat> Fs;

		for (int i = 0; i < nviews; i++)
		{
			const int viewIndex = viewIndices.at<int>(i);
			Is.push_back(images[viewIndex]);
			Ks.push_back(cameras[viewIndex]);
			Rs.push_back(rotations[viewIndex]);
			ts.push_back(translations[viewIndex]);
			Ds.push_back(depthMaps[viewIndex]);
			Fs.push_back(features[viewIndex]);
		}

		auto& image = images[referenceIdx];
		auto& depths = depthMaps[referenceIdx];
		auto& normals = normalMaps[referenceIdx];

		/////////////////////////////////////////////////////////////////////////////////////////////////////
		// random initialization
		/////////////////////////////////////////////////////////////////////////////////////////////////////

		// project to camera coordinate
		std::vector<Vec3f> Xcs;

		// compute depth range
		float minZ, maxZ;
		if (hasDepthRange_)
		{
			// the depth range is given explicitly, the sparse points are not used
			minZ = depthRangeMin_;
			maxZ = depthRangeMax_;
		}
		else
		{
			projectToCamera(sparsePoints, Rs[0], ts[0], Xcs);
			if (Xcs.empty())
				CV_Error(Error::StsBadArg, "The initial object points must not be empty unless an explicit depth range is set");

			std::sort(std::begin(Xcs), std::end(Xcs), [](const Vec3f& lhs, const Vec3f& rhs) { return lhs[2] < rhs[2]; });
			minZ = floatCast(LAMBDA_MIN * Xcs[cvFloor(C_MIN * Xcs.size())][2]);
			maxZ = floatCast(LAMBDA_MAX * Xcs[cvFloor(C_MAX * Xcs.size())][2]);
		}

		std::vector<TextureImage> d_Fs, d_Ds;
		uploadImages(Fs, d_Fs);
		uploadHomographyParams(Ks, Rs, ts);
		if (geom)
			uploadDepths(Ds, d_Ds);

		GpuMat d_depths = createContinuous(image.size(), CV_32F);
		GpuMat d_normals = createContinuous(image.size(), CV_32FC3);
		GpuMat d_costs = createContinuous(image.size(), CV_32F);

		if (randomInit)
		{
			gpu::initializeDepthsAndNormals(d_depths, d_normals, minZ, maxZ);
		}
		else
		{
			d_depths.upload(depths);
			d_normals.upload(normals);
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////
		// calculate initial matching scores
		/////////////////////////////////////////////////////////////////////////////////////////////////////

		// create propagation method
		auto propagation = createPropagation(Fs, Ks, Rs, ts, Ds, minZ, maxZ, geom);

		propagation->calcInitialCosts(d_depths, d_normals, d_costs);
		DEBUG_CALLBACK(image, d_depths, d_normals, d_costs, "Initial depth and costs");

		/////////////////////////////////////////////////////////////////////////////////////////////////////
		// propagate and refine
		/////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int iter = 0; iter < pmIters_; iter++)
		{
			propagation->propagateAndRefine(d_depths, d_normals, d_costs, PIXEL_COLOR_BLK);
			DEBUG_CALLBACK(image, d_depths, d_normals, d_costs, cv::format("Black propagation at iter %d", iter + 1));

			propagation->propagateAndRefine(d_depths, d_normals, d_costs, PIXEL_COLOR_RED);
			DEBUG_CALLBACK(image, d_depths, d_normals, d_costs, cv::format("Red propagation at iter %d", iter + 1));

			propagation->next();
		}

		d_depths.download(depths);
		d_normals.download(normals);
	}

	void compute(InputArrayOfArrays _images, InputArrayOfArrays _cameras, InputArrayOfArrays _rotations,
		InputArrayOfArrays _translations, InputArrayOfArrays _viewIdSets, InputArray _initialObjectPoints,
		OutputArray _objectPoints, OutputArray _objectColors, OutputArray _objectNormals, OutputArrayOfArrays _depths) override
	{
		// convert input arrays to mat vector
		std::vector<Mat> images, cameras, rotations, translations, viewIdSets;
		getInputMatVectors(_images, _cameras, _rotations, _translations, _viewIdSets,
			images, cameras, rotations, translations, viewIdSets);

		const Mat sparsePoints = _initialObjectPoints.getMat();
		if (!sparsePoints.empty())
			CV_Assert(sparsePoints.type() == CV_32FC3 || sparsePoints.type() == CV_64FC3);

		const int nimages = static_cast<int>(images.size());
		const int nreferences = static_cast<int>(viewIdSets.size());
		const auto activeViewIndices = getActiveViewIndices(viewIdSets);

		std::vector<Mat> depths(nimages), normals(nimages), masks(nimages);
		std::vector<int> levels(nimages, 0);
		const int maxLevel = multiScale_ ? calcPyramidLevels(images, levels) : 0;

		for (int s = 0; s <= maxLevel; s++)
		{
			std::vector<Mat> featureImages(nimages);
			std::vector<Mat> scaledCameras(nimages);
			std::vector<Mat> scaledImages(nimages);

			for (int i = 0; i < nimages; i++)
			{
				if (!activeViewIndices.count(i))
					continue;

				const double scale = (multiScale_ && levels[i] > 0) ? 1. / (1 << levels[i]) : 1;
				scaledImages[i] = scale < 1 ? scaleImage(images[i], scale) : images[i];
				scaledCameras[i] = scale < 1 ? scaleCameraParams(cameras[i], scale) : cameras[i];
				featureTransform(scaledImages[i], featureImages[i]);

				if (s != 0)
				{
					resize(depths[i], depths[i], scaledImages[i].size(), 0, 0, INTER_NEAREST);
					resize(normals[i], normals[i], scaledImages[i].size(), 0, 0, INTER_NEAREST);
				}

				levels[i]--;
			}

			for (int iter = 0; iter < 1 + gcIters_; iter++)
			{
				const bool randomInit = s == 0 && iter == 0;
				const bool geom = iter > 0;
				for (int i = 0; i < nreferences; i++)
				{
					const int referenceIdx = viewIdSets[i].at<int>(0);

					DEBUG_PRINT("[MVS INFO] processing image %3d / %3d, ", i + 1, nreferences);
					DEBUG_PRINT("reference view: %3d geometric consistency: %d\n", referenceIdx, geom);

					estimateDepth(scaledImages, scaledCameras, rotations, translations, featureImages,
						referenceIdx, viewIdSets[i], sparsePoints, depths, normals, geom, randomInit);
				}
			}
		}
		for (int i = 0; i < nreferences; i++)
		{
			const int referenceIdx = viewIdSets[i].at<int>(0);
			masks[referenceIdx] = Mat::ones(depths[referenceIdx].size(), CV_8U);
		}

		// generate point cloud
		std::vector<Vec3f> objectPoints;
		std::vector<Vec3b> objectColors;
		std::vector<Vec3f> objectNormals;
		for (int i = 0; i < nreferences; i++)
		{
			const int referenceIdx = viewIdSets[i].at<int>(0);

			DEBUG_PRINT("[MVS INFO] processing image %3d / %3d, ", i + 1, nreferences);
			DEBUG_PRINT("reference view: %3d\n", referenceIdx);

			fuseDepthMaps(images, cameras, rotations, translations, viewIdSets[i],
				depths, normals, masks, objectPoints, objectColors, objectNormals);
		}

		// an empty cloud cannot be reshaped to one row, so the outputs are
		// cleared instead
		if (objectPoints.empty())
		{
			if (_objectPoints.needed()) _objectPoints.release();
			if (_objectColors.needed()) _objectColors.release();
			if (_objectNormals.needed()) _objectNormals.release();
		}
		else
		{
			Mat(objectPoints).reshape(3, 1).copyTo(_objectPoints);
			Mat(objectColors).reshape(3, 1).copyTo(_objectColors);
			Mat(objectNormals).reshape(3, 1).copyTo(_objectNormals);
		}

		if (_depths.needed())
		{
			CV_Assert(_depths.isMatVector());
			_depths.create(nimages, 1, CV_32F);
			for (int i = 0; i < nimages; i++)
			{
				_depths.create(depths[i].size(), CV_32F, i);
				depths[i].copyTo(_depths.getMat(i));
			}
		}
	}

	void setPatchMatchMethod(PatchMatchMethod method) override
	{
		method_ = method;
	}

	void setPatchMatchIters(int pmIters) override
	{
		pmIters_ = pmIters;
	}

	void setGeometricConsistencyIters(int gcIters) override
	{
		gcIters_ = gcIters;
	}

	void setMultiScale(bool multiScale) override
	{
		multiScale_ = multiScale;
	}

	void setDebugCallBack(DebugCallBack callBack) override
	{
		callBack_ = callBack;
	}

	void setDepthRange(double minZ, double maxZ) override
	{
		if (!(minZ > 0.0))
			CV_Error(Error::StsBadArg, "The minimum depth must be positive");
		if (!(maxZ > minZ))
			CV_Error(Error::StsBadArg, "The maximum depth must be larger than the minimum depth");
		depthRangeMin_ = floatCast(minZ);
		depthRangeMax_ = floatCast(maxZ);
		hasDepthRange_ = true;
	}

private:

	PatchMatchMethod method_;
	int pmIters_;
	int gcIters_;
	bool multiScale_;
	DebugCallBack callBack_;
	bool hasDepthRange_;
	float depthRangeMin_;
	float depthRangeMax_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// API
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Ptr<PatchMatchMVS> PatchMatchMVS::create(PatchMatchMethod method, int pmIters, int gcIters, bool multiScale)
{
	return makePtr<PatchMatchMVSImpl>(method, pmIters, gcIters, multiScale);
}

PatchMatchMVS::~PatchMatchMVS()
{
}

} // namespace cuda
} // namespace cv
