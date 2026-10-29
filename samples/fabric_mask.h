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

#ifndef __FABRIC_MASK_H__
#define __FABRIC_MASK_H__

#include <opencv2/core.hpp>

namespace cv
{

/** @brief Options of the fabric mask estimation.

The mask is estimated from a temporal brightness maximum: in a dark room the
projector is the only light source, so the white fabric is the only large bright
region, and taking the per-pixel maximum over several frames fills the regions
that are dark in the projected content of a single frame.
*/
struct FabricMaskOptions
{
	int frames = 10;          //!< number of frames accumulated into the brightness maximum
	double threshold = -1.0;  //!< brightness threshold in [0, 255], negative: Otsu
	double minArea = 0.02;    //!< minimum fabric area as a fraction of the image
	int closeSize = 9;        //!< size of the morphological closing kernel [px]
	int openSize = 5;         //!< size of the morphological opening kernel [px]
	int erodeSize = 3;        //!< size of the erosion kernel [px], 0: no erosion
};

/** @brief Accumulates the per-pixel brightness maximum of a frame.
@param maxGray The accumulated maximum (CV_8UC1).  An empty matrix is
initialized with the first frame.
@param frame The next frame (color or grayscale).
*/
void accumulateFabricBrightness(Mat& maxGray, const Mat& frame);

/** @brief Estimates the mask of the fabric from the temporal brightness maximum.
The bright fabric is separated with Otsu (or a manual threshold), small holes are
closed, small bright blobs (reflections, markers, LEDs) are removed, all large
components are filled and the result is eroded to stay inside the fabric.
@param maxGray The temporal brightness maximum (CV_8UC1).
@param options The estimation options.
@return A CV_8UC1 mask (255 inside the fabric, 0 outside).  The mask is empty if
no fabric was found, e.g. because it is smaller than #FabricMaskOptions::minArea.
*/
Mat estimateFabricMask(const Mat& maxGray, const FabricMaskOptions& options);

} // namespace cv

#endif // !__FABRIC_MASK_H__
