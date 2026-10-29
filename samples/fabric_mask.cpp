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

#include "fabric_mask.h"

#include <algorithm>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace cv
{

void accumulateFabricBrightness(Mat& maxGray, const Mat& frame)
{
	Mat gray;
	if (frame.channels() == 3)
		cvtColor(frame, gray, COLOR_BGR2GRAY);
	else if (frame.channels() == 4)
		cvtColor(frame, gray, COLOR_BGRA2GRAY);
	else
		gray = frame;

	if (maxGray.empty())
	{
		gray.copyTo(maxGray);
		return;
	}

	CV_Assert(maxGray.type() == CV_8UC1 && maxGray.size() == gray.size());
	cv::max(maxGray, gray, maxGray);
}

Mat estimateFabricMask(const Mat& maxGray, const FabricMaskOptions& options)
{
	CV_Assert(maxGray.type() == CV_8UC1);
	if (maxGray.empty())
		return Mat();

	// the projector speckle and the sensor noise are high frequency; a small
	// blur keeps the fabric boundary while removing single hot pixels
	Mat blurred;
	GaussianBlur(maxGray, blurred, Size(5, 5), 0);

	Mat binary;
	if (options.threshold > 0.0)
	{
		cv::threshold(blurred, binary, options.threshold, 255, THRESH_BINARY);
	}
	else
	{
		// Otsu separates the lit fabric from the dark room; the temporal
		// maximum makes the fabric bright in all its pixels, so Otsu does not
		// split the fabric at dark parts of the projected content
		cv::threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	}

	if (countNonZero(binary) == 0)
		return Mat();

	// close the small dark holes that are never bright in any frame and remove
	// small bright blobs such as reflections, LEDs or ArUco markers
	const int closeSize = std::max(1, options.closeSize | 1);
	const int openSize = std::max(1, options.openSize | 1);
	Mat closed, opened;
	morphologyEx(binary, closed, MORPH_CLOSE,
		getStructuringElement(MORPH_RECT, Size(closeSize, closeSize)));
	morphologyEx(closed, opened, MORPH_OPEN,
		getStructuringElement(MORPH_RECT, Size(openSize, openSize)));

	std::vector<std::vector<Point>> contours;
	findContours(opened, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (contours.empty())
		return Mat();

	double largest = 0.0;
	for (const auto& contour : contours)
		largest = std::max(largest, contourArea(contour));

	const double imageArea = static_cast<double>(maxGray.total());
	if (largest <= 0.0 || largest < options.minArea * imageArea)
		return Mat();

	// keep the largest component and all components of comparable size; the
	// fabric can be split into several parts by a dark band of the projection
	const double minComponent = std::max(0.2 * largest, 0.001 * imageArea);
	Mat mask = Mat::zeros(maxGray.size(), CV_8U);
	for (const auto& contour : contours)
	{
		if (contourArea(contour) < minComponent)
			continue;

		// filling the external contour also fills its holes
		drawContours(mask, std::vector<std::vector<Point>>{ contour }, 0, Scalar(255), FILLED);
	}

	if (countNonZero(mask) == 0)
		return Mat();

	// stay inside the fabric: the boundary pixels are a mixture of fabric and
	// background, and a slightly smaller mask loses only a thin border
	if (options.erodeSize > 0)
	{
		const int erodeSize = std::max(1, options.erodeSize | 1);
		erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(erodeSize, erodeSize)));
		if (countNonZero(mask) == 0)
			return Mat();
	}

	return mask;
}

} // namespace cv
