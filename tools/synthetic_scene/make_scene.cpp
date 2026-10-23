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

// Synthetic multi-view scene generator for end-to-end verification of the
// cuda-multi-view-stereo pipeline inside the Docker image.
//
// Builds a classic stereo rig (N cameras translated along +x, identity
// rotation) looking at a slightly tilted, randomly textured plane.  Writes the
// exact same FileStorage/PLY layout that samples/app_initialize_ETH3D produces,
// so app_patch_match_mvs can consume it unmodified.
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv)
{
	const std::string outDir = (argc > 1) ? argv[1] : "/tmp/scene";
	std::filesystem::create_directories(outDir);

	const int W = 400, H = 300;          // even width (propagation is a checkerboard)
	const double fx = 400.0, fy = 400.0, cx = 200.0, cy = 150.0;
	const double planeBase = 4.0;        // plane: z = planeBase + tilt * x
	const double tilt = 0.1;
	const double baseline = 0.1;         // camera translation step along x
	const int N = 4;                     // number of views
	const int TEX = 1024;                // texture resolution
	const double texel = 0.01;           // world units per texel

	// ---- texture: natural 1/f^2 image spectrum -------------------------------
	// Pure per-pixel noise is pathological for the bilateral ZNCC cost (almost
	// no pixel pair has a similar colour, so no support survives the
	// weighting).
	//
	// The matching cost is a *bilateral* ZNCC: the weight of a sample is
	// exp(-(I - Icenter)^2 / (2 * 0.2^2)), i.e. it only keeps reference pixels
	// whose intensity is close to the patch centre.  A texture with strong
	// local gradients (e.g. 8 px random blobs) therefore leaves a 9x9 patch
	// with almost no support and every match degenerates.
	//
	// A natural 1/f^2 image spectrum is used instead: locally smooth (so the
	// bilateral weight keeps most samples) yet globally distinctive (so the
	// 9x9 patch is still unique enough to match).
	cv::Mat tex(TEX, TEX, CV_32FC3);
	{
		std::mt19937 rng(20240607);
		std::uniform_real_distribution<double> u01(0.0, 1.0);
		const double lambdas[] = { 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128 };
		const int nOct = static_cast<int>(sizeof(lambdas) / sizeof(lambdas[0]));

		struct Wave { double a, kx, ky, phi; };
		std::vector<Wave> waves;
		double norm = 0.0;
		for (int c = 0; c < 3; ++c)
		{
			for (int o = 0; o < nOct; ++o)
			{
				const double lam = lambdas[o];
				const double amp = 1.0 / lam;           // 1/f amplitude  => 1/f^2 power
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

		for (int y = 0; y < TEX; ++y)
		{
			for (int x = 0; x < TEX; ++x)
			{
				cv::Vec3f v(0.f, 0.f, 0.f);
				for (size_t i = 0; i < waves.size(); ++i)
				{
					const Wave& w = waves[i];
					const float s = static_cast<float>(w.a * std::sin(w.kx * x + w.ky * y + w.phi));
					v[0] += s; v[1] += s; v[2] += s;
				}
				// per-channel offset so the three channels are not identical
				tex.at<cv::Vec3f>(y, x) = v * static_cast<float>(1.0 / norm);
			}
		}
		// normalise each channel to [0, 1]
		std::vector<cv::Mat> chans;
		cv::split(tex, chans);
		for (auto& ch : chans)
		{
			double lo, hi;
			cv::minMaxLoc(ch, &lo, &hi);
			ch = (ch - lo) * (1.0 / std::max(hi - lo, 1e-9));
		}
		cv::merge(chans, tex);
		tex.convertTo(tex, CV_8UC3, 255.0);
	}

	// ---- cameras -------------------------------------------------------------
	std::vector<cv::Mat> Ks, Rs, ts;
	std::vector<std::string> names;

	for (int i = 0; i < N; ++i)
	{
		const double camX = i * baseline;

		cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
			0.0, fy, cy,
			0.0, 0.0, 1.0);
		const cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
		// The library's pose convention (see app_initialize_ETH3D.cpp) is
		//     X_world = R * X_cam + t
		// i.e. R is the camera-to-world rotation and t is the camera centre in
		// world coordinates.  With an identity orientation the centre is simply
		// (camX, 0, 0).
		const cv::Mat t = (cv::Mat_<double>(1, 3) << camX, 0.0, 0.0);

		Ks.push_back(K);
		Rs.push_back(R);
		ts.push_back(t);

		// ---- render by ray casting onto the tilted plane ---------------------
		cv::Mat img(H, W, CV_8UC3);
		// K = [[fx,0,cx],[0,fy,cy],[0,0,1]]  =>  explicit inverse
		const cv::Matx33d Ki(1.0 / fx, 0.0, -cx / fx,
			0.0, 1.0 / fy, -cy / fy,
			0.0, 0.0, 1.0);
		for (int v = 0; v < H; ++v)
		{
			for (int u = 0; u < W; ++u)
			{
				const cv::Vec3d d = Ki * cv::Vec3d(u, v, 1.0);
				const double denom = d[2] - tilt * d[0];
				if (std::abs(denom) < 1e-9) { img.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0); continue; }
				const double s = (planeBase + tilt * camX) / denom;
				const double px = camX + s * d[0];
				const double py = s * d[1];

				int ix = static_cast<int>(std::lround((px + 2.0) / texel));
				int iy = static_cast<int>(std::lround((py + 2.0) / texel));
				ix = std::clamp(ix, 0, TEX - 1);
				iy = std::clamp(iy, 0, TEX - 1);
				img.at<cv::Vec3b>(v, u) = tex.at<cv::Vec3b>(iy, ix);
			}
		}

		const std::string path = outDir + "/view" + std::to_string(i) + ".png";
		cv::imwrite(path, img);
		names.push_back(path);
	}

	// ---- input_images.json ---------------------------------------------------
	{
		cv::FileStorage fs(outDir + "/input_images.json", cv::FileStorage::WRITE);
		fs << "num_images" << N;
		fs << "input_images" << "[";
		for (int i = 0; i < N; ++i)
		{
			fs << "{:";
			fs << "id" << i;
			fs << "filename" << names[i];
			fs << "K" << Ks[i].reshape(1, 3);
			fs << "}";
		}
		fs << "]";
	}

	// ---- global_poses.json ---------------------------------------------------
	{
		cv::FileStorage fs(outDir + "/global_poses.json", cv::FileStorage::WRITE);
		fs << "num_global_poses" << N;
		fs << "global_poses" << "[";
		for (int i = 0; i < N; ++i)
		{
			fs << "{:";
			fs << "id" << i;
			fs << "R" << Rs[i].reshape(1, 3);
			fs << "t" << ts[i].reshape(1, 3);
			fs << "}";
		}
		fs << "]";
	}

	// ---- view_id_sets.json ---------------------------------------------------
	// viewIdSets[i] = [referenceViewIdx, n1, n2, ...]  (first entry MUST be the reference)
	{
		std::vector<cv::Mat> viewIdSets;
		for (int ref = 0; ref < N; ++ref)
		{
			std::vector<int> ids{ ref };
			for (int other = 0; other < N; ++other)
			{
				if (other != ref) { ids.push_back(other); }
			}
			cv::Mat m(static_cast<int>(ids.size()), 1, CV_32S);
			for (size_t k = 0; k < ids.size(); ++k) { m.at<int>(static_cast<int>(k)) = ids[k]; }
			viewIdSets.push_back(m);
		}

		cv::FileStorage fs(outDir + "/view_id_sets.json", cv::FileStorage::WRITE);
		fs << "num_reference_images" << static_cast<int>(viewIdSets.size());
		fs << "view_id_sets" << "[";
		for (const auto& v : viewIdSets)
		{
			fs << "{:";
			fs << "id" << v.at<int>(0);
			fs << "view_id_set" << v;
			fs << "}";
		}
		fs << "]";
	}

	// ---- point_cloud_sparse.ply ----------------------------------------------
	{
		std::mt19937 rng2(777);
		std::uniform_real_distribution<double> ux(-1.8, 1.8), uy(-1.3, 1.3);
		cv::Mat points(400, 1, CV_32FC3);
		cv::Mat colors(400, 1, CV_8UC3);
		for (int i = 0; i < 400; ++i)
		{
			const double px = ux(rng2), py = uy(rng2);
			points.at<cv::Vec3f>(i) = cv::Vec3f(
				static_cast<float>(px),
				static_cast<float>(py),
				static_cast<float>(planeBase + tilt * px));
			colors.at<cv::Vec3b>(i) = cv::Vec3b(
				static_cast<uint8_t>(rng2() & 0xFF),
				static_cast<uint8_t>(rng2() & 0xFF),
				static_cast<uint8_t>(rng2() & 0xFF));
		}
		cv::viz::writeCloud(outDir + "/point_cloud_sparse.ply", points, colors);
	}

	std::cout << "synthetic scene written to " << outDir << std::endl;
	std::cout << "  views     : " << N << " (" << W << "x" << H << ")" << std::endl;
	std::cout << "  baseline  : " << baseline << " world units" << std::endl;
	std::cout << "  plane     : z = " << planeBase << " + " << tilt << " * x" << std::endl;
	return 0;
}
