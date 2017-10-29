/**
 *
 * A program to test a LSTM Neural Network
 * Author: Brandon Trabucco
 * Date: 2016/07/27
 *
 */

#include "LSTMNetwork.cuh"
#include "DatasetAdapter.h"
#include "OutputTarget.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

long long getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

struct tm *getDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

int main(int argc, char *argv[]) {
	cout << "Program initializing" << endl;
	if (argc < 5) {
		cout << argv[0] << " <learning rate> <decay rate> <blocks> <cells> <size ...>" << endl;
		return -1;
	}

	int updatePoints = 10;
	int savePoints = 10;
	int maxEpoch = 10;
	int trainingSize = 500;
	int blocks = atoi(argv[3]);
	int cells = atoi(argv[4]);
	int sumNeurons = (blocks * cells);
	double errorBound = 0.01;
	double mse = 0;
	double learningRate = atof(argv[1]), decayRate = atof(argv[2]);
	long long networkStart, networkEnd, sumTime = 0, iterationStart;

	const int _day = getDate()->tm_mday;


	/**
	 *
	 * 	Open file streams to save data samples from Neural Network
	 * 	This data can be plotted via GNUPlot
	 *
	 */
	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/Temporal_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Multicore-LSTM-Error_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream errorData(errorDataFileName.str(), ios::app);
	if (!errorData.is_open()) return -1;


	ostringstream accuracyDataFileName;
	accuracyDataFileName << "/u/trabucco/Desktop/Temporal_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_GPU-LSTM-Accuracy_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream accuracyData(accuracyDataFileName.str(), ios::app);
	if (!accuracyData.is_open()) return -1;

	ostringstream timingDataFileName;
	timingDataFileName << "/u/trabucco/Desktop/Sequential_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_GPU-LSTM-Timing_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream timingData(timingDataFileName.str(), ios::app);
	if (!timingData.is_open()) return -1;

	ostringstream outputDataFileName;
	outputDataFileName << "/u/trabucco/Desktop/Sequential_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_GPU-LSTM-Output_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream outputData(outputDataFileName.str(), ios::app);
	if (!outputData.is_open()) return -1;
	outputData << endl << endl;


	networkStart = getMSec();
	DatasetAdapter dataset = DatasetAdapter();
	networkEnd = getMSec();
	cout << "Language Dataset loaded in " << (networkEnd - networkStart) << "msecs" << endl;


	LSTMNetwork network = LSTMNetwork(dataset.getCharSize(), blocks, cells, learningRate, decayRate);
	OutputTarget target = OutputTarget(dataset.getCharSize(), dataset.getCharSize());
	cout << "Network initialized" << endl;


	for (int i = 0; i < (argc - 5); i++) {
		network.addLayer(atoi(argv[5 + i]));
		sumNeurons += atoi(argv[5 + i]);
	} network.addLayer(dataset.getCharSize());


	int totalIterations = 0;
	bool converged = false;
	for (int e = 0; (e < maxEpoch)/* && (!e || (((mse1 + mse2)/2) > errorBound))*/; e++) {
		int c = 0, n = 0;
		vector<double> error, output;

		networkStart = getMSec();
		for (int i = 0; i < trainingSize && dataset.nextChar(); i++) {
			DatasetExample data = dataset.getChar();
			error = network.train(target.getOutputFromTarget(data.current),
					target.getOutputFromTarget(data.next));
		}

		dataset.reset();

		for (int i = 0; i < trainingSize && dataset.nextChar(); i++) {
			DatasetExample data = dataset.getChar();
			output = network.classify(target.getOutputFromTarget(data.current));

			n++;
			if (target.getTargetFromOutput(output) == (int)data.next) c++;
		} networkEnd = getMSec();

		sumTime += (networkEnd - networkStart);
		totalIterations += 1;

		mse = 0;
		for (int i = 0; i < error.size(); i++)
			mse += error[i] * error[i];
		mse /= error.size() * 2;

		if (((e + 1) % (maxEpoch / updatePoints)) == 0) {
			cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
			cout << "Error[" << e << "] = " << mse << endl;
			cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)n) << endl;
		} errorData << e << ", " << mse << endl;
		accuracyData << e << ", " << (100.0 * (float)c / (float)n) << endl;

		dataset.reset();
	}

	vector<vector<double> > seed;
	seed.push_back(target.getOutputFromTarget((int)'I'));
	for (int i = 0; i < 500; i++) {
		vector<double> output = network.classify(seed[i]);
		seed.push_back(output);
		char text = (char)target.getTargetFromOutput(output);
		outputData << text;
	}

	timingData << sumNeurons << ", " << sumTime << ", " << totalIterations << endl;
	timingData.close();
	accuracyData.close();
	errorData.close();
	outputData.close();

	return 0;
}
