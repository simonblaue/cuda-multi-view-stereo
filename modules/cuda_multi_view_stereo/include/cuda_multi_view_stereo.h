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

#ifndef __CUDA_MULTI_VIEW_STEREO_H__
#define __CUDA_MULTI_VIEW_STEREO_H__

#include <opencv2/core.hpp>

namespace cv
{
namespace cuda
{

/** @brief The PatchMatch methods
*/
enum PatchMatchMethod
{
	GIPUMA_FAST, //!< Gipuma, fast settings
	ACMH,        //!< Adaptive Checkerboard sampling and Multi-Hypothesis joint view selection
};

/** @brief The class implements PatchMatch-based multi-view stereo.
*/
class PatchMatchMVS
{
public:

	/** @brief The callback function for debugging and visualization
	*/
	using DebugCallBack = void(*)(InputArray image, InputArray depths, InputArray normals, InputArray costs, const String& title);

	/** @brief The PatchMatchMVS constructor
	@param method PatchMatch method, see #PatchMatchMethod.
	@param pmIters Number of innier PatchMatch iterations.
	@param gcIters Number of outer PatchMatch iterations performed with geometric consistency.
	@param multiScale If true, the algorithm uses multi-scale method
	*/
	static Ptr<PatchMatchMVS> create(PatchMatchMethod method = ACMH, int pmIters = 3, int gcIters = 0, bool multiScale = false);

	/** @brief The destructor
	*/
	virtual ~PatchMatchMVS();

	/** @brief Computes depth maps for the given images and reprojects to 3D space.
	@param images Input array of color images.
	@param cameras Input array of camera matrices.
	@param rotations Input array of 3x3 rotation matrices.
	@param translations Input array of 3D translation vectors.
	@param viewIdSets Input array of "viewIdSet"s,
	a viewIdSet consists of [ reference view index, neighbor view index #1, neighbor view index #2, ... ]
	@param initialObjectPoints Iutput array of sparse 3D points in object coordinates.
	@param objectPoints Output array of densified 3D points in object coordinates.
	@param objectColors Output array of colors of the same size with objectPoints.
	@param objectNormals Output array of normals of the same size with objectPoints.
	@param depths Output array of depth maps.
	*/
	virtual void compute(InputArrayOfArrays images, InputArrayOfArrays cameras, InputArrayOfArrays rotations,
		InputArrayOfArrays translations, InputArrayOfArrays viewIdSets, InputArray initialObjectPoints,
		OutputArray objectPoints, OutputArray objectColors, OutputArray objectNormals,
		OutputArrayOfArrays depths = noArray()) = 0;

	virtual void setPatchMatchMethod(PatchMatchMethod method) = 0;
	virtual void setPatchMatchIters(int pmIters) = 0;
	virtual void setGeometricConsistencyIters(int gcIters) = 0;
	virtual void setMultiScale(bool multiScale) = 0;
	virtual void setDebugCallBack(DebugCallBack callBack) = 0;

	/** @brief Sets the depth range used by the PatchMatch search explicitly.
	@param minZ Minimum depth (distance along the optical axis of the reference camera).
	@param maxZ Maximum depth.
	By default the depth range is derived from the initial object points (the 1%
	and 99% percentiles of their depths, scaled by 0.75 and 1.25).  When the
	range is set explicitly, the initial object points are not used any more
	and may be empty.  This is useful for multi-view setups without a sparse
	point cloud, e.g. video sequences with a known calibration only.
	*/
	virtual void setDepthRange(double minZ, double maxZ) = 0;
};

} // namespace cuda
} // namespace cv

#endif // !__CUDA_MULTI_VIEW_STEREO_H__
