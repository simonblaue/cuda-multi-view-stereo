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
Generates a synthetic synchronised multi-camera video sequence together with a
rig calibration file, so that app_video_mvs can be verified without real data.

The scene is a tilted plane  z = 4 + 0.1 * x  covered by a natural 1/f^2 image
spectrum texture.  The cameras are translated along +x and keep their identity
orientation, so the rig is a set of parallel stereo pairs.  The lens distortion
of the calibration is applied while rendering, so the undistortion path of
app_video_mvs is exercised as well.

    make_video_scene <output-directory> [--cameras=4] [--frames=10] ...

Output:
    <output-directory>/camera_0.mp4 ...
    <output-directory>/calibration.yml
*/

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

static std::string keys =
"{ @output-directory | /tmp/video_scene | directory the videos are written to.                     }"
"{ cameras          |         4 | number of cameras of the rig.                                    }"
"{ frames           |        10 | number of frames of the sequence.                                }"
"{ width            |       640 | width of the rendered images.                                    }"
"{ height           |       480 | height of the rendered images.                                   }"
"{ baseline         |       0.3 | distance between two neighbouring cameras.                       }"
"{ scene-step       |      0.02 | displacement of the scene between two frames.                    }"
"{ distortion       |         1 | whether the images are rendered with lens distortion.           }"
"{ help h           |            | print help message.                                             }";

namespace
{

// a natural 1/f^2 image spectrum: locally smooth (the bilateral ZNCC of the
// matching cost needs support) yet globally distinctive.
cv::Mat makeTexture(int size, unsigned int seed)
{
	const double lambdas[] = { 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128 };
	const int nOct = static_cast<int>(sizeof(lambdas) / sizeof(lambdas[0]));

	struct Wave { double a, kx, ky, phi; };
	std::vector<Wave> waves;
	double norm = 0.0;

	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> u01(0.0, 1.0);

	for (int c = 0; c < 3; ++c)
	{
		for (int o = 0; o < nOct; ++o)
		{
			const double lam = lambdas[o];
			const double amp = 1.0 / lam;
			const double th = 2.0 * CV_PI * u01(rng);

			Wave w;
			w.a = amp;
			w.kx = 2.0 * CV_PI * std::cos(th) / lam;
			w.ky = 2.0 * CV_PI * std::sin(th) / lam;
			w.phi = 2.0 * CV_PI * u01(rng);

			waves.push_back(w);
			norm += amp * amp;
		}
	}
	norm = std::sqrt(norm / 3.0);

	cv::Mat texture(size, size, CV_32FC3);
	for (int y = 0; y < size; ++y)
	{
		for (int x = 0; x < size; ++x)
		{
			cv::Vec3f v(0.f, 0.f, 0.f);
			for (const auto& w : waves)
			{
				const float s = static_cast<float>(w.a * std::sin(w.kx * x + w.ky * y + w.phi));
				v[0] += s; v[1] += s; v[2] += s;
			}
			texture.at<cv::Vec3f>(y, x) = v * static_cast<float>(1.0 / norm);
		}
	}

	std::vector<cv::Mat> channels;
	cv::split(texture, channels);
	for (auto& channel : channels)
	{
		double lo = 0.0, hi = 0.0;
		cv::minMaxLoc(channel, &lo, &hi);
		channel = (channel - lo) * (1.0 / std::max(hi - lo, 1e-9));
	}
	cv::merge(channels, texture);
	texture.convertTo(texture, CV_8UC3, 255.0);
	return texture;
}

/** @brief Bilinearly samples the 8-bit texture and returns the colour in [0, 255].
@param texture The texture image (CV_8UC3).
@param tx The x coordinate in texels.
@param ty The y coordinate in texels.
@return The interpolated colour.
*/
cv::Vec3f sampleTexture(const cv::Mat& texture, double tx, double ty)
{
	const int x0 = static_cast<int>(std::floor(tx));
	const int y0 = static_cast<int>(std::floor(ty));
	const float ax = static_cast<float>(tx - x0);
	const float ay = static_cast<float>(ty - y0);

	cv::Vec3f color(0.f, 0.f, 0.f);
	for (int dy = 0; dy < 2; ++dy)
	{
		for (int dx = 0; dx < 2; ++dx)
		{
			const int x = std::min(std::max(x0 + dx, 0), texture.cols - 1);
			const int y = std::min(std::max(y0 + dy, 0), texture.rows - 1);
			const float w = (dx ? ax : 1.f - ax) * (dy ? ay : 1.f - ay);
			const cv::Vec3b texel = texture.at<cv::Vec3b>(y, x);
			color += w * cv::Vec3f(static_cast<float>(texel[0]), static_cast<float>(texel[1]),
				static_cast<float>(texel[2]));
		}
	}
	return color;
}

// plumb-bob distortion model with five coefficients
cv::Vec2d distort(double x, double y, const cv::Mat& distortion)
{
	const double k1 = distortion.at<double>(0);
	const double k2 = distortion.at<double>(1);
	const double p1 = distortion.at<double>(2);
	const double p2 = distortion.at<double>(3);
	const double k3 = distortion.at<double>(4);

	const double r2 = x * x + y * y;
	const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;

	return cv::Vec2d(x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
		y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);
}

// inverse of distort: a physical lens maps a scene ray r to the distorted
// pixel K * distort(r), so rendering pixel (u, v) means undistorting its
// normalized coordinates; fixed point iteration on the residual converges
// quickly for mild distortion
cv::Vec2d undistortPoint(double xd, double yd, const cv::Mat& distortion)
{
	cv::Vec2d p(xd, yd);
	for (int iter = 0; iter < 20; iter++)
	{
		const cv::Vec2d d = distort(p[0], p[1], distortion);
		p += cv::Vec2d(xd, yd) - d;
	}
	return p;
}

} // namespace

int main(int argc, char* argv[])
{
	using namespace cv;

	const CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	const auto outputDir = parser.get<std::string>("@output-directory");
	const auto ncameras = parser.get<int>("cameras");
	const auto nframes = parser.get<int>("frames");
	const auto width = parser.get<int>("width");
	const auto height = parser.get<int>("height");
	const auto baseline = parser.get<double>("baseline");
	const auto sceneStep = parser.get<double>("scene-step");
	const auto useDistortion = parser.get<int>("distortion") == 1;

	if (!parser.check())
	{
		parser.printErrors();
		parser.printMessage();
		return EXIT_FAILURE;
	}

	CV_Assert(ncameras >= 2);
	CV_Assert(nframes >= 1);

	std::filesystem::create_directories(outputDir);

	// scene
	const double planeBase = 4.0;
	const double tilt = 0.1;
	const double texel = 0.01;
	const int textureSize = 1024;
	const cv::Mat texture = makeTexture(textureSize, 20240607);

	// cameras: identity orientation, translated along +x
	const double fx = width, fy = height;
	const double cx = width * 0.5, cy = height * 0.5;

	const Mat K = (Mat_<double>(3, 3) << fx, 0.0, cx,
		0.0, fy, cy,
		0.0, 0.0, 1.0);

	Mat distortion = (Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
	if (useDistortion)
		distortion = (Mat_<double>(1, 5) << -0.08, 0.02, 0.001, -0.001, 0.0);

	// write the calibration
	{
		const String path = outputDir + "/calibration.yml";
		FileStorage fs(path, FileStorage::WRITE);
		fs << "nb_camera" << ncameras;
		for (int i = 0; i < ncameras; i++)
		{
			const Mat pose = (Mat_<double>(4, 4) << 1.0, 0.0, 0.0, i * baseline,
				0.0, 1.0, 0.0, 0.0,
				0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 1.0);

			fs << format("camera_%d", i) << "{:";
			fs << "camera_matrix" << K;
			fs << "distortion_vector" << distortion;
			fs << "distortion_type" << 0;
			fs << "camera_group" << 0;
			fs << "img_width" << width;
			fs << "img_height" << height;
			fs << "camera_pose_matrix" << pose;
			fs << "}";
		}
		std::cout << "calibration written to " << path << std::endl;
	}

	// render the videos
	for (int i = 0; i < ncameras; i++)
	{
		const String path = format("%s/camera_%d.mp4", outputDir.c_str(), i);

		VideoWriter writer;
		if (!writer.open(path, VideoWriter::fourcc('m', 'p', '4', 'v'), 10.0, Size(width, height), true))
		{
			std::cerr << "cannot open the video writer for " << path << std::endl;
			return EXIT_FAILURE;
		}

		const double camX = i * baseline;

		for (int f = 0; f < nframes; f++)
		{
			const double shift = f * sceneStep;

			Mat image(height, width, CV_8UC3);
			for (int v = 0; v < height; v++)
			{
				for (int u = 0; u < width; u++)
				{
					const double x = (u - cx) / fx;
					const double y = (v - cy) / fy;

					// the ray of the pixel: undistort the normalized coordinates
					const Vec2d d = useDistortion ? undistortPoint(x, y, distortion) : Vec2d(x, y);

					// intersected with the plane
					const double s = (planeBase - tilt * shift + tilt * camX) / (1.0 - tilt * d[0]);

					const double px = camX + s * d[0];
					const double py = s * d[1];

					const Vec3f color = sampleTexture(texture, (px + 5.0) / texel, (py + 5.0) / texel);
					image.at<Vec3b>(v, u) = Vec3b(
						static_cast<uchar>(color[2]),
						static_cast<uchar>(color[1]),
						static_cast<uchar>(color[0]));
				}
			}

			writer.write(image);
		}

		writer.release();
		std::cout << "video written to " << path << std::endl;
	}

	std::cout << std::endl;
	std::cout << "synthetic video scene written to " << outputDir << std::endl;
	std::cout << "  cameras    : " << ncameras << " (" << width << "x" << height << ")" << std::endl;
	std::cout << "  frames     : " << nframes << std::endl;
	std::cout << "  baseline   : " << baseline << " world units" << std::endl;
	std::cout << "  plane      : z = " << planeBase << " + " << tilt << " * x" << std::endl;
	std::cout << "  distortion : " << (useDistortion ? "yes" : "no") << std::endl;

	return 0;
}
