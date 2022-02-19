#pragma once

#include <random>
#include <vector>

namespace dle {

	using std::vector;

	class Dataset {
	public:
		static void makeDataset(
			int start, int end, double mu1, double m2, int answer,
			vector<vector<double>>& x, vector<int>& t,
			std::mt19937& rng);

	private:
		Dataset();
	};
}
