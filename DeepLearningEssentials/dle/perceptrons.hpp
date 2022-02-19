#pragma once

#include <vector>

namespace dle {

	using std::vector;

	class Perceptrons {
	public:
		Perceptrons(int nIn);
		virtual ~Perceptrons() {}

		int train(const vector<double>& x, int t, double learningRate);
		int predict(const vector<double>& x) const;

	private:
		int nIn_;
		vector<double> w_;
	};

}