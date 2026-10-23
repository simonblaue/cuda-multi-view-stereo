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

#include "rig_plot.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace cv
{
namespace
{

// drawing constants
const int kImageWidth = 1800;
const int kMargin = 30;
const int kPanelWidth = 560;
const int kPanelHeight = 520;
const int kPanelGap = 30;
const int kTitleHeight = 90;
const int kTableRowHeight = 26;

// colours
const Scalar kBackground(255, 255, 255);
const Scalar kPanelBackground(248, 248, 250);
const Scalar kPanelBorder(200, 200, 208);
const Scalar kGrid(228, 228, 234);
const Scalar kAxis(170, 170, 180);
const Scalar kText(35, 35, 35);
const Scalar kSubText(115, 115, 122);
const Scalar kBaseline(140, 140, 148);
const Scalar kLabelBackground(255, 255, 255);

const std::vector<Scalar>& cameraPalette()
{
	static const std::vector<Scalar> palette = {
		Scalar(70, 90, 225),   // blue
		Scalar(80, 170, 90),   // green
		Scalar(60, 140, 240),  // orange
		Scalar(185, 90, 175),  // purple
		Scalar(60, 60, 60),    // dark grey
		Scalar(40, 190, 225),  // yellow
		Scalar(85, 60, 200),   // red
		Scalar(130, 160, 60),  // teal
	};
	return palette;
}

String formatNumber(double value, int precision)
{
	std::ostringstream stream;
	stream << std::fixed << std::setprecision(precision) << value;
	return stream.str();
}

/** @brief A two dimensional coordinate plot of the world frame.
*/
struct Plot2D
{
	Vec3d u{1.0, 0.0, 0.0};  //!< world direction of the horizontal axis
	Vec3d v{0.0, 0.0, 1.0};  //!< world direction of the vertical axis (up on the image)
	Rect rect;               //!< area of the plot in the image
	double scale = 1.0;      //!< pixels per metre
	Point2d origin;          //!< position of the world origin in the image
	double xmin = 0.0;       //!< left edge of the plot in world coordinates
	double xmax = 0.0;       //!< right edge of the plot in world coordinates
	double ymin = 0.0;       //!< bottom edge of the plot in world coordinates
	double ymax = 0.0;       //!< top edge of the plot in world coordinates
	double width = 0.0;      //!< width of the content in world coordinates
	double height = 0.0;     //!< height of the content in world coordinates
};

/** @brief Projects a point of the world frame into the image.
*/
Point2d project(const Plot2D& plot, const Vec3d& point)
{
	return Point2d(plot.origin.x + plot.scale * point.dot(plot.u),
		plot.origin.y - plot.scale * point.dot(plot.v));
}

/** @brief Builds a coordinate plot that contains all the given points.
@param rect The area of the plot in the image.
@param u The world direction of the horizontal axis.
@param v The world direction of the vertical axis.
@param points The points of the world frame that have to be visible.
@param margin The free space around the points, relative to the size of the plot.
*/
Plot2D makePlot(const Rect& rect, const Vec3d& u, const Vec3d& v, const std::vector<Vec3d>& points, double margin)
{
	Plot2D plot;
	plot.u = u;
	plot.v = v;
	plot.rect = rect;

	double minX = 0.0, maxX = 0.0, minY = 0.0, maxY = 0.0;
	for (const auto& point : points)
	{
		const double x = point.dot(u);
		const double y = point.dot(v);
		minX = std::min(minX, x);
		maxX = std::max(maxX, x);
		minY = std::min(minY, y);
		maxY = std::max(maxY, y);
	}

	const double rangeX = std::max(maxX - minX, 1e-3);
	const double rangeY = std::max(maxY - minY, 1e-3);
	const double availableWidth = rect.width * (1.0 - 2.0 * margin);
	const double availableHeight = rect.height * (1.0 - 2.0 * margin);

	plot.scale = std::min(availableWidth / rangeX, availableHeight / rangeY);
	if (!(plot.scale > 0.0) || !std::isfinite(plot.scale))
		plot.scale = 1.0;

	const double centerX = 0.5 * (minX + maxX);
	const double centerY = 0.5 * (minY + maxY);
	plot.origin = Point2d(rect.x + 0.5 * rect.width - plot.scale * centerX,
		rect.y + 0.5 * rect.height + plot.scale * centerY);

	plot.xmin = (rect.x - plot.origin.x) / plot.scale;
	plot.xmax = (rect.x + rect.width - plot.origin.x) / plot.scale;
	plot.ymax = (plot.origin.y - rect.y) / plot.scale;
	plot.ymin = (plot.origin.y - rect.y - rect.height) / plot.scale;
	plot.width = rangeX;
	plot.height = rangeY;

	return plot;
}

/** @brief Returns a round grid step that gives at most ten lines.
*/
double niceStep(double range)
{
	const double candidates[] = { 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0 };
	for (const double candidate : candidates)
	{
		if (range / candidate <= 10.0)
			return candidate;
	}
	return 100.0;
}

/** @brief Draws the grid, the axes and the ticks of a coordinate plot.
*/
void drawGrid(Mat& image, const Plot2D& plot, const String& uName, const String& vName)
{
	const double step = niceStep(std::max(plot.xmax - plot.xmin, plot.ymax - plot.ymin));
	const int decimals = (step < 0.1) ? 2 : ((step < 1.0) ? 1 : 0);

	// vertical lines (constant coordinate of the horizontal axis)
	const int nx = static_cast<int>(std::floor(plot.xmax / step)) - static_cast<int>(std::ceil(plot.xmin / step)) + 1;
	for (int i = 0; i < nx; i++)
	{
		const double x = (std::ceil(plot.xmin / step) + i) * step;
		const double px = plot.origin.x + plot.scale * x;
		const bool isAxis = std::abs(x) < 1e-9;
		line(image, Point2d(px, plot.rect.y), Point2d(px, plot.rect.y + plot.rect.height),
			isAxis ? kAxis : kGrid, 1, LINE_AA);
		putText(image, formatNumber(x, decimals), Point(cvRound(px) + 3, plot.rect.y + plot.rect.height - 6),
			FONT_HERSHEY_SIMPLEX, 0.36, kSubText, 1, LINE_AA);
	}

	// horizontal lines (constant coordinate of the vertical axis)
	const int ny = static_cast<int>(std::floor(plot.ymax / step)) - static_cast<int>(std::ceil(plot.ymin / step)) + 1;
	for (int i = 0; i < ny; i++)
	{
		const double y = (std::ceil(plot.ymin / step) + i) * step;
		const double py = plot.origin.y - plot.scale * y;
		const bool isAxis = std::abs(y) < 1e-9;
		line(image, Point2d(plot.rect.x, py), Point2d(plot.rect.x + plot.rect.width, py),
			isAxis ? kAxis : kGrid, 1, LINE_AA);
		putText(image, formatNumber(y, decimals), Point(plot.rect.x + 4, cvRound(py) - 4),
			FONT_HERSHEY_SIMPLEX, 0.36, kSubText, 1, LINE_AA);
	}

	// axis indicators
	const Point indicator(plot.rect.x + plot.rect.width - 62, plot.rect.y + plot.rect.height - 26);
	arrowedLine(image, indicator, indicator + Point(38, 0), kText, 1, LINE_AA, 0.2);
	putText(image, uName, indicator + Point(42, 4), FONT_HERSHEY_SIMPLEX, 0.42, kText, 1, LINE_AA);
	arrowedLine(image, indicator, indicator + Point(0, -38), kText, 1, LINE_AA, 0.2);
	putText(image, vName, indicator + Point(-8, -42), FONT_HERSHEY_SIMPLEX, 0.42, kText, 1, LINE_AA);
}

/** @brief Returns the box of a label drawn at the given position.
*/
Rect labelBox(const String& text, const Point2d& position, double fontScale)
{
	int baseline = 0;
	const Size size = getTextSize(text, FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline);
	return Rect(cvRound(position.x) - 2, cvRound(position.y) - size.height - 1,
		size.width + 4, size.height + baseline + 2);
}

/** @brief Draws a text with a white background box.
*/
void drawLabel(Mat& image, const String& text, const Point2d& position, double fontScale, const Scalar& color)
{
	const Rect box = labelBox(text, position, fontScale);
	rectangle(image, box, kLabelBackground, FILLED);
	putText(image, text, Point(cvRound(position.x), cvRound(position.y)), FONT_HERSHEY_SIMPLEX, fontScale, color, 1, LINE_AA);
}

/** @brief Returns the intersection of two boxes, an empty box if they do not overlap.
*/
Rect intersect(const Rect& a, const Rect& b)
{
	const Rect overlap = a & b;
	return (overlap.width > 0 && overlap.height > 0) ? overlap : Rect();
}

/** @brief Draws the header of a table and returns the position of the first row.
*/
int drawTableHeader(Mat& image, const Rect& area, const String& title, const std::vector<String>& columns,
	const std::vector<int>& offsets)
{
	putText(image, title, Point(area.x, area.y), FONT_HERSHEY_SIMPLEX, 0.58, kText, 1, LINE_AA);
	line(image, Point(area.x, area.y + 8), Point(area.x + area.width, area.y + 8), kPanelBorder, 1, LINE_AA);

	int y = area.y + 34;
	for (size_t i = 0; i < columns.size(); i++)
		putText(image, columns[i], Point(area.x + offsets[i], y), FONT_HERSHEY_SIMPLEX, 0.46, kSubText, 1, LINE_AA);
	line(image, Point(area.x, y + 6), Point(area.x + area.width, y + 6), kGrid, 1, LINE_AA);

	return y + kTableRowHeight;
}

} // namespace

Vec3d opticalAxis(const Mat& rotation)
{
	const Matx33d R = rotation.reshape(1, 3);
	return Vec3d(R(0, 2), R(1, 2), R(2, 2));
}

std::vector<RigPairMeasurement> measureRig(const std::vector<RigCamera>& cameras,
	const std::vector<std::pair<int, int>>& pairs)
{
	std::vector<RigPairMeasurement> measurements;
	measurements.reserve(pairs.size());

	for (const auto& pair : pairs)
	{
		const int i = pair.first;
		const int j = pair.second;
		if (i < 0 || j < 0 || i >= static_cast<int>(cameras.size()) || j >= static_cast<int>(cameras.size()))
			continue;

		const Vec3d& c0 = cameras[i].center;
		const Vec3d& c1 = cameras[j].center;
		const Vec3d& a0 = cameras[i].axis;
		const Vec3d& a1 = cameras[j].axis;

		const Vec3d delta = c1 - c0;
		const double distance = std::sqrt(delta.dot(delta));
		const double horizontal = std::sqrt(delta[0] * delta[0] + delta[2] * delta[2]);

		RigPairMeasurement measurement;
		measurement.camera0 = i;
		measurement.camera1 = j;
		measurement.distance = distance;
		measurement.horizontal = horizontal;
		measurement.height = delta[1];
		measurement.axisAngle = std::acos(std::min(1.0, std::max(-1.0, a0.dot(a1)))) * 180.0 / CV_PI;

		// the yaw angle is the angle between the horizontal projections of the
		// optical axes
		const double yaw0 = std::atan2(a0[0], a0[2]);
		const double yaw1 = std::atan2(a1[0], a1[2]);
		double yaw = std::abs(yaw0 - yaw1) * 180.0 / CV_PI;
		while (yaw > 180.0)
			yaw -= 360.0;
		measurement.yawAngle = yaw;

		// the triangulation angle is the angle between the optical axis of the
		// first camera and the baseline
		if (distance > 1e-9)
		{
			const Vec3d direction = delta / distance;
			measurement.triangulationAngle =
				std::acos(std::min(1.0, std::max(-1.0, a0.dot(direction)))) * 180.0 / CV_PI;
		}

		measurements.push_back(measurement);
	}

	return measurements;
}

void printRigMeasurements(const std::vector<RigCamera>& cameras, const std::vector<RigPairMeasurement>& measurements)
{
	std::cout << "measurements between the neighbouring cameras:" << std::endl;
	std::cout << "  pair  distance [m]  horizontal [m]  height [m]  axis angle [deg]  yaw angle [deg]  triangulation [deg]"
		<< std::endl;
	for (const auto& measurement : measurements)
	{
		std::cout << format("  %d-%d    %10.4f    %13.4f    %10.4f    %14.3f    %13.3f    %19.3f\n",
			measurement.camera0, measurement.camera1, measurement.distance, measurement.horizontal,
			measurement.height, measurement.axisAngle, measurement.yawAngle, measurement.triangulationAngle).c_str();
	}
	std::cout << std::endl;

	for (size_t i = 0; i < cameras.size(); i++)
	{
		const Vec3d& axis = cameras[i].axis;
		// the azimuth is measured from the Z axis towards the X axis and the
		// elevation is positive when the camera looks up (the Y axis of the
		// camera convention points down)
		const double azimuth = std::atan2(axis[0], axis[2]) * 180.0 / CV_PI;
		const double elevation = -std::asin(std::min(1.0, std::max(-1.0, axis[1]))) * 180.0 / CV_PI;
		std::cout << format("  camera_%d: centre (%.4f, %.4f, %.4f), azimuth %.2f deg, elevation %.2f deg, hFOV %.2f deg\n",
			static_cast<int>(i), cameras[i].center[0], cameras[i].center[1], cameras[i].center[2],
			azimuth, elevation, cameras[i].horizontalFov).c_str();
	}
	std::cout << std::endl;
}

void drawRigLayout(const std::vector<RigCamera>& cameras, const std::vector<std::pair<int, int>>& pairs,
	const std::vector<RigPairMeasurement>& measurements, const std::string& title,
	const std::string& subtitle, Mat& image)
{
	const int nrows = static_cast<int>(measurements.size()) + static_cast<int>(cameras.size()) + 4;
	const int tableHeight = 40 + nrows * kTableRowHeight;
	const int imageHeight = kTitleHeight + kPanelHeight + 70 + tableHeight + kMargin;

	image = Mat(imageHeight, kImageWidth, CV_8UC3, kBackground);

	// title
	putText(image, title, Point(kMargin, 44), FONT_HERSHEY_SIMPLEX, 0.95, kText, 2, LINE_AA);
	putText(image, subtitle, Point(kMargin, 70), FONT_HERSHEY_SIMPLEX, 0.48, kSubText, 1, LINE_AA);

	// the camera centres of the rig
	std::vector<Vec3d> centers;
	centers.reserve(cameras.size());
	for (const auto& camera : cameras)
		centers.push_back(camera.center);

	const Rect panelRect(kMargin, kTitleHeight, kPanelWidth, kPanelHeight);
	const Rect frontRect(kMargin + kPanelWidth + kPanelGap, kTitleHeight, kPanelWidth, kPanelHeight);
	const Rect sideRect(kMargin + 2 * (kPanelWidth + kPanelGap), kTitleHeight, kPanelWidth, kPanelHeight);

	// the plots are built twice: the second build takes the length of the
	// optical axis arrows into account, so they stay inside the plots
	double margin = 0.16;
	Plot2D topPlot = makePlot(panelRect, Vec3d(1.0, 0.0, 0.0), Vec3d(0.0, 0.0, 1.0), centers, margin);
	Plot2D frontPlot = makePlot(frontRect, Vec3d(1.0, 0.0, 0.0), Vec3d(0.0, 1.0, 0.0), centers, margin);
	Plot2D sidePlot = makePlot(sideRect, Vec3d(0.0, 0.0, 1.0), Vec3d(0.0, 1.0, 0.0), centers, margin);

	double contentRadius = 0.5 * std::max(topPlot.width, topPlot.height) * topPlot.scale;
	double axisLength = std::min(120.0, std::max(45.0, 0.35 * contentRadius));

	margin = std::max(margin, (axisLength + 20.0) / std::min(kPanelWidth, kPanelHeight));
	topPlot = makePlot(panelRect, Vec3d(1.0, 0.0, 0.0), Vec3d(0.0, 0.0, 1.0), centers, margin);
	frontPlot = makePlot(frontRect, Vec3d(1.0, 0.0, 0.0), Vec3d(0.0, 1.0, 0.0), centers, margin);
	sidePlot = makePlot(sideRect, Vec3d(0.0, 0.0, 1.0), Vec3d(0.0, 1.0, 0.0), centers, margin);

	contentRadius = 0.5 * std::max(topPlot.width, topPlot.height) * topPlot.scale;
	axisLength = std::min(120.0, std::max(45.0, 0.35 * contentRadius));

	const Plot2D* plots[] = { &topPlot, &frontPlot, &sidePlot };
	const String plotNames[] = { "X-Z plane (top view)", "X-Y plane (front view)", "Z-Y plane (side view)" };
	const String uNames[] = { "X", "X", "Z" };
	const String vNames[] = { "Z", "Y", "Y" };

	for (int p = 0; p < 3; p++)
	{
		const Plot2D& plot = *plots[p];
		rectangle(image, plot.rect, kPanelBackground, FILLED);
		rectangle(image, plot.rect, kPanelBorder, 1);
		putText(image, plotNames[p], Point(plot.rect.x + 12, plot.rect.y + 26), FONT_HERSHEY_SIMPLEX, 0.55, kText, 1, LINE_AA);
	}

	// the field of view cones are only drawn in the top view because the
	// horizontal field of view is the meaningful one for the layout of the rig
	// the cones are drawn in a separate layer, so they cannot leave the plot
	{
		Mat layer = image(topPlot.rect).clone();
		for (size_t i = 0; i < cameras.size(); i++)
		{
			const Vec3d& axis = cameras[i].axis;
			const double norm = std::sqrt(axis[0] * axis[0] + axis[2] * axis[2]);
			if (norm < 1e-6 || !(cameras[i].horizontalFov > 0.0) || !std::isfinite(cameras[i].horizontalFov))
				continue;

			const double dx = axis[0] / norm;
			const double dy = -axis[2] / norm;
			// the cone has to stay convex, so the half angle is limited
			const double halfAngle = std::min(cameras[i].horizontalFov * 0.5, 80.0) * CV_PI / 180.0;
			const double length = 2.0 * axisLength;

			const Point2d center = project(topPlot, cameras[i].center);
			std::vector<Point> polygon;
			polygon.push_back(Point(cvRound(center.x) - topPlot.rect.x, cvRound(center.y) - topPlot.rect.y));
			const int narc = 24;
			for (int k = 0; k <= narc; k++)
			{
				const double angle = -halfAngle + 2.0 * halfAngle * k / narc;
				const double ux = std::cos(angle) * dx - std::sin(angle) * dy;
				const double uy = std::sin(angle) * dx + std::cos(angle) * dy;
				polygon.push_back(Point(cvRound(center.x - topPlot.rect.x + length * ux),
					cvRound(center.y - topPlot.rect.y - length * uy)));
			}

			const Scalar color = cameraPalette()[i % cameraPalette().size()];
			const Scalar light(kPanelBackground + 0.16 * (color - kPanelBackground));
			fillConvexPoly(layer, polygon, light, LINE_AA);
		}
		layer.copyTo(image(topPlot.rect));
	}

	// grid of the plots
	drawGrid(image, topPlot, uNames[0], vNames[0]);
	drawGrid(image, frontPlot, uNames[1], vNames[1]);
	drawGrid(image, sidePlot, uNames[2], vNames[2]);

	// baselines of the neighbouring cameras
	for (const auto& pair : pairs)
	{
		const int i = pair.first;
		const int j = pair.second;
		if (i < 0 || j < 0 || i >= static_cast<int>(cameras.size()) || j >= static_cast<int>(cameras.size()))
			continue;

		for (int p = 0; p < 3; p++)
		{
			const Plot2D& plot = *plots[p];
			const Point2d p0 = project(plot, cameras[i].center);
			const Point2d p1 = project(plot, cameras[j].center);
			line(image, p0, p1, kBaseline, 1, LINE_AA);
		}
	}

	// cameras and optical axes
	for (size_t i = 0; i < cameras.size(); i++)
	{
		const Scalar color = cameraPalette()[i % cameraPalette().size()];

		for (int p = 0; p < 3; p++)
		{
			const Plot2D& plot = *plots[p];
			const Point2d center = project(plot, cameras[i].center);
			const Point2d point(cvRound(center.x), cvRound(center.y));

			// the optical axis
			const double dx = cameras[i].axis.dot(plot.u);
			const double dy = -cameras[i].axis.dot(plot.v);
			const double norm = std::sqrt(dx * dx + dy * dy);
			if (norm > 1e-6)
			{
				const Point2d tip(center.x + axisLength * dx / norm, center.y + axisLength * dy / norm);
				arrowedLine(image, point, tip, color, 2, LINE_AA, 0.15);
			}

			// the camera
			circle(image, point, 9, kBackground, FILLED, LINE_AA);
			circle(image, point, 9, color, 2, LINE_AA);
			putText(image, format("%d", static_cast<int>(i)), Point(cvRound(point.x) - 5, cvRound(point.y) + 5),
				FONT_HERSHEY_SIMPLEX, 0.45, color, 1, LINE_AA);
		}
	}

	// the labels of the baselines hold the distance between the camera centres
	// and the angle between the optical axes; they are placed next to the
	// baseline they belong to, without overlapping each other
	{
		// the cameras and the labels that are already placed are obstacles
		std::vector<Rect> obstacles;
		for (size_t i = 0; i < cameras.size(); i++)
		{
			const Point2d center = project(topPlot, cameras[i].center);
			obstacles.push_back(Rect(cvRound(center.x) - 11, cvRound(center.y) - 11, 22, 22));
		}

		for (const auto& pair : pairs)
		{
			const int i = pair.first;
			const int j = pair.second;
			if (i < 0 || j < 0 || i >= static_cast<int>(cameras.size()) || j >= static_cast<int>(cameras.size()))
				continue;

			const RigPairMeasurement* measurement = nullptr;
			for (const auto& m : measurements)
			{
				if (m.camera0 == i && m.camera1 == j)
				{
					measurement = &m;
					break;
				}
			}
			if (measurement == nullptr)
				continue;

			const String text = format("%s / %s",
				formatNumber(measurement->distance, 3).c_str(), formatNumber(measurement->axisAngle, 1).c_str());

			const Point2d p0 = project(topPlot, cameras[i].center);
			const Point2d p1 = project(topPlot, cameras[j].center);
			const Point2d anchor = 0.5 * (p0 + p1);
			Point2d direction = p1 - p0;
			const double length = std::sqrt(direction.dot(direction));
			if (length < 1e-6)
				continue;
			direction *= 1.0 / length;
			// the label is moved away from the baseline
			const Point2d normal(-direction.y, direction.x);

			// the label is moved along the normal of the baseline until it does
			// not overlap another label or a camera any more and until it stays
			// inside the plot
			Point2d position(0.0, 0.0);
			Rect box;
			bool placed = false;
			for (const double offset : { 14.0, 28.0, 42.0, 56.0, -14.0, -28.0, -42.0, -56.0 })
			{
				position = anchor + normal * offset;
				box = labelBox(text, position, 0.38);
				if (box.x < topPlot.rect.x + 2 || box.y < topPlot.rect.y + 2 ||
					box.br().x > topPlot.rect.br().x - 2 || box.br().y > topPlot.rect.br().y - 2)
					continue;

				bool overlap = false;
				for (const auto& obstacle : obstacles)
				{
					if (intersect(box, obstacle).area() > 0)
					{
						overlap = true;
						break;
					}
				}
				if (!overlap)
				{
					placed = true;
					break;
				}
			}
			if (!placed)
				position = anchor + normal * 14.0;

			// the label is kept inside the plot
			Rect bounds = labelBox(text, position, 0.38);
			if (bounds.x < topPlot.rect.x + 2)
				position.x += topPlot.rect.x + 2 - bounds.x;
			if (bounds.y < topPlot.rect.y + 2)
				position.y += topPlot.rect.y + 2 - bounds.y;
			bounds = labelBox(text, position, 0.38);
			if (bounds.br().x > topPlot.rect.br().x - 2)
				position.x -= bounds.br().x - (topPlot.rect.br().x - 2);
			if (bounds.br().y > topPlot.rect.br().y - 2)
				position.y -= bounds.br().y - (topPlot.rect.br().y - 2);

			drawLabel(image, text, position, 0.38, kText);
			obstacles.push_back(labelBox(text, position, 0.38));
		}
	}

	// scale bar of the top view
	{
		const double step = niceStep(topPlot.width);
		const double length = step * topPlot.scale;
		const Point2d origin(topPlot.rect.x + 20, topPlot.rect.y + topPlot.rect.height - 24);
		line(image, origin, origin + Point2d(length, 0.0), kText, 2, LINE_AA);
		line(image, origin + Point2d(0, -5), origin + Point2d(0, 5), kText, 2, LINE_AA);
		line(image, origin + Point2d(length, -5), origin + Point2d(length, 5), kText, 2, LINE_AA);
		putText(image, format("%s m", formatNumber(step, step < 1.0 ? 1 : 0).c_str()),
			Point(cvRound(origin.x), cvRound(origin.y) - 10), FONT_HERSHEY_SIMPLEX, 0.42, kText, 1, LINE_AA);
	}

	// legend
	putText(image, "baseline label: distance between the camera centres [m] / angle between the optical axes [deg]",
		Point(kMargin, kTitleHeight + kPanelHeight + 28), FONT_HERSHEY_SIMPLEX, 0.44, kSubText, 1, LINE_AA);
	putText(image, "the plots show the world axes of the calibration in metres, the vertical axis of every plot points up",
		Point(kMargin, kTitleHeight + kPanelHeight + 48), FONT_HERSHEY_SIMPLEX, 0.44, kSubText, 1, LINE_AA);

	// table of the measurements between the neighbouring cameras
	const Rect pairArea(kMargin, kTitleHeight + kPanelHeight + 70, 1030, tableHeight);
	const std::vector<String> pairColumns = {
		"pair", "distance [m]", "horizontal [m]", "height [m]", "axis angle [deg]", "yaw angle [deg]", "triangulation [deg]"
	};
	const std::vector<int> pairOffsets = { 0, 80, 240, 400, 560, 730, 880 };
	int y = drawTableHeader(image, pairArea, "Neighbouring cameras", pairColumns, pairOffsets);

	for (const auto& measurement : measurements)
	{
		const String values[] = {
			format("%d-%d", measurement.camera0, measurement.camera1),
			formatNumber(measurement.distance, 4),
			formatNumber(measurement.horizontal, 4),
			formatNumber(measurement.height, 4),
			formatNumber(measurement.axisAngle, 2),
			formatNumber(measurement.yawAngle, 2),
			formatNumber(measurement.triangulationAngle, 2)
		};
		for (size_t c = 0; c < 7; c++)
			putText(image, values[c], Point(pairArea.x + pairOffsets[c], y), FONT_HERSHEY_SIMPLEX, 0.46, kText, 1, LINE_AA);
		y += kTableRowHeight;
	}

	// table of the cameras
	const Rect cameraArea(kMargin + 1060, kTitleHeight + kPanelHeight + 70, 700, tableHeight);
	const std::vector<String> cameraColumns = {
		"camera", "X [m]", "Y [m]", "Z [m]", "azimuth [deg]", "elevation [deg]", "hFOV [deg]"
	};
	const std::vector<int> cameraOffsets = { 0, 85, 185, 285, 385, 500, 620 };
	y = drawTableHeader(image, cameraArea, "Cameras", cameraColumns, cameraOffsets);

	for (size_t i = 0; i < cameras.size(); i++)
	{
		const Vec3d& axis = cameras[i].axis;
		const double azimuth = std::atan2(axis[0], axis[2]) * 180.0 / CV_PI;
		const double elevation = -std::asin(std::min(1.0, std::max(-1.0, axis[1]))) * 180.0 / CV_PI;

		const String values[] = {
			format("camera_%d", static_cast<int>(i)),
			formatNumber(cameras[i].center[0], 4),
			formatNumber(cameras[i].center[1], 4),
			formatNumber(cameras[i].center[2], 4),
			formatNumber(azimuth, 2),
			formatNumber(elevation, 2),
			formatNumber(cameras[i].horizontalFov, 2)
		};
		for (size_t c = 0; c < 7; c++)
			putText(image, values[c], Point(cameraArea.x + cameraOffsets[c], y), FONT_HERSHEY_SIMPLEX, 0.46, kText, 1, LINE_AA);
		y += kTableRowHeight;
	}
}

} // namespace cv
