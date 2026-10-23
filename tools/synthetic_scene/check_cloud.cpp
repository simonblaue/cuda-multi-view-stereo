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

// Reads the dense PLY produced by app_patch_match_mvs and compares the
// reconstructed z against the ground-truth plane z = 4 + 0.1 * x.
//
// The PLY is parsed directly (ascii) because viz::readCloud mis-reports the
// vertex count for this file layout.
#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
	const std::string path = (argc > 1) ? argv[1] : "/tmp/out/point_cloud_dense.ply";
	std::ifstream in(path);
	if (!in) { printf("cannot open %s\n", path.c_str()); return 1; }

	std::string line;
	int nvertices = -1;
	while (std::getline(in, line))
	{
		if (line.rfind("element vertex", 0) == 0)
		{
			nvertices = std::stoi(line.substr(std::string("element vertex").size()));
			break;
		}
	}
	if (nvertices < 0) { printf("no vertex element in %s\n", path.c_str()); return 1; }

	std::vector<double> err, zs, xs;
	while (std::getline(in, line))
	{
		if (line.empty() || line[0] == 'p' || line[0] == 'e' || line[0] == 'f') continue;
		std::istringstream ss(line);
		double x, y, z;
		int r, g, b;
		if (!(ss >> x >> y >> z >> r >> g >> b)) continue;
		xs.push_back(x);
		zs.push_back(z);
		err.push_back(std::abs(z - (4.0 + 0.1 * x)));
	}
	if (err.empty()) { printf("no vertices parsed\n"); return 1; }

	std::sort(err.begin(), err.end());
	std::sort(zs.begin(), zs.end());
	std::sort(xs.begin(), xs.end());
	const double med = err[err.size() / 2];
	const double p90 = err[(size_t)(err.size() * 0.9)];
	const double p99 = err[(size_t)(err.size() * 0.99)];

	printf("dense points        : %d\n", (int)err.size());
	printf("median |dz|         : %.4f\n", med);
	printf("p90    |dz|         : %.4f\n", p90);
	printf("p99    |dz|         : %.4f\n", p99);
	printf("z range             : [%.3f, %.3f]  (ground truth 3.8 .. 4.2)\n", zs.front(), zs.back());
	printf("x range             : [%.3f, %.3f]\n", xs.front(), xs.back());
	printf("fraction |dz|<0.1   : %.1f %%\n", 100.0 * (double)std::count_if(err.begin(), err.end(),
		[](double e) { return e < 0.1; }) / (double)err.size());
	return 0;
}
