#include "test_perceptrons.hpp"
#include "perceptrons.hpp"
#include "dataset.hpp"

#include <random>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::to_string;

namespace dle {

	void testPerceptrons() {

		cout << "set consigs" << endl;

		std::random_device rd;
		std::mt19937 rng(rd());
		
		const int trainN = 1000; // number of training data
		const int trainSize = static_cast<int>(trainN / 2); // TODO rename

		const int testN = 1000; // number of test data
		const int testSize = static_cast<int>(testN / 2); // TODO rename
		
		const int nIn = 2; // dim of input data
		// const int nOut = 1;

		const int epochs = 100;
		const double learningRate = 1.; // learning rate can be 1 in perceptrons

		cout << "initialize tensors" << endl;

		vector<vector<double>> trainX(trainN, vector<double>(nIn, 0.)); // input data for training
		vector<int> trainT(trainN, 0); // answers (labels) for training

		vector<vector<double>> testX(testN, vector<double>(nIn, 0.)); // input data for test
		vector<int> testT(testN, 0); // answers (labels) for test
		vector<int> predictedT(testN, 0); // outputs predicted by the model
		
		cout << "make dataset" << endl;

		// class1 inputs x11 and x12: x11 ~ N(-2.0, 1.0), x12 ~ N(+2.0, 1.0)
		const double mu11 = -2.;
		const double mu12 = 2.;
		const int answer1 = 1;
		// make training data for class1
		Dataset::makeDataset(0, trainSize, mu11, mu12, answer1, trainX, trainT, rng);
		// make test data for class1
		Dataset::makeDataset(0, testSize, mu11, mu12, answer1, testX, testT, rng);

		// class2 inputs x21 and x22: x21 ~ N(+2.0, 1.0), x22 ~ N(-2.0, 1.0)
		const double mu21 = 2.;
		const double mu22 = -2.;
		const int answer2 = -1;
		// make training data for class2
		Dataset::makeDataset(trainSize, trainN, mu21, mu22, answer2, trainX, trainT, rng);
		// make test data for class2
		Dataset::makeDataset(testSize, testN, mu21, mu22, answer2, testX, testT, rng);

		// build the model

		// construct
		Perceptrons classifier(nIn);

		// train
		cout << "train" << endl;
		int epoch = 0; // training epoch counter
		while (true) {
			cout << "epoch: " << to_string(epoch) << endl;

			int classified = 0;
			for (int i = 0; i < trainN; ++i)
				classified += classifier.train(trainX[i], trainT[i], learningRate);

			cout << "number of data classified correctly = " << to_string(classified) << endl;

			if (classified == trainN) // when all data are classified correctly
				break;

			epoch += 1;
			if (epoch > epochs)
				break;
		}

		// test
		cout << "test" << endl;
		for (int i = 0; i < testN; ++i)
			predictedT[i] = classifier.predict(testX[i]);

		// evaluate the model
		cout << "evaluate the model" << endl;
		vector<vector<int>> confusionMatrix(2, vector<int>(2, 0));
		double accuracy = 0.;
		double precision = 0.;
		double recall = 0.;
		for (int i = 0; i < testN; ++i) {
			if (predictedT[i] > 0) { // positive
				if (testT[i] > 0) { // TP
					accuracy += 1;
					precision += 1;
					recall += 1;
					confusionMatrix[0][0] += 1;
				}
				else { // FP
					confusionMatrix[1][0] += 1;
				}
			}
			else { // negative
				if (testT[i] > 0) { // FN
					confusionMatrix[0][1] += 1;
				}
				else { // TN
					accuracy += 1;
					confusionMatrix[1][1] += 1;
				}
			}
		}

		accuracy /= testN;

		const int nPredictedPositive = confusionMatrix[0][0] + confusionMatrix[1][0];
		precision /= static_cast<double>(nPredictedPositive);
		
		const int nRealPositive = confusionMatrix[0][0] + confusionMatrix[0][1];
		recall /= static_cast<double>(nRealPositive);

		cout << "Perceptrons model evaluation" << endl;
		cout << "Accuracy: " << to_string(accuracy * 100.) << endl;
		cout << "Precision: " << to_string(precision * 100.) << endl;
		cout << "Recall: " << to_string(recall * 100.) << endl;
	}

}