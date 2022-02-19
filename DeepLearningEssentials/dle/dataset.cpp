#include "dataset.hpp"

#include <iostream>
#include <string>

namespace dle {

	void Dataset::makeDataset(
		int start, int end, double mu1, double mu2, int answer,
		vector<vector<double>>& x, vector<int>& t,
		std::mt19937& rng) {

		std::normal_distribution<double> normal(0., 1.);

		for (int i = start; i < end; ++i) {
			x[i][0] = normal(rng) + mu1;
			x[i][1] = normal(rng) + mu2;
			t[i] = answer;
		}
	}

}