#include "perceptrons.hpp"
#include "activation_function.hpp"

#include <iostream>
#include <string>

using std::to_string;

namespace dle {

	Perceptrons::Perceptrons(int nIn): nIn_(nIn), w_(vector<double>(nIn)) {}

	int Perceptrons::train(const vector<double>& x, int t, double learningRate) {
		
		// check if the data is classified correctly
		double c = 0.;
		for (int i = 0; i < nIn_; ++i)
			c += w_[i] * x[i] * t;

		// apply steepest descent method if the data is wrongly classified
		int classified = 0;
		if (c > 0.) // correct
			classified = 1;
		else { // wrong
			for (int i = 0; i < nIn_; ++i)
				w_[i] += learningRate * x[i] * t;
			std::cout << "updated w[0]: " << to_string(w_[0])
				<< ", w[1]: " << to_string(w_[1]) << std::endl;
		}

		return classified;
	}

	int Perceptrons::predict(const vector<double>& x) const {
		
		double preActivation = 0.;
		for (int i = 0; i < nIn_; ++i)
			preActivation += w_[i] * x[i];
		
		return activation::step(preActivation);
	}
}