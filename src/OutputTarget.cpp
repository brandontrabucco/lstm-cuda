/*
 * OutputTarget.cpp
 *
 *  Created on: Jun 23, 2016
 *      Author: trabucco
 */

#include "OutputTarget.h"

OutputTarget::OutputTarget(int n, int c) {
	nodes = n;
	classes = c;

	for (int i = 0; i < c; i++) {
		vector<double> temp;
		for (int j = 0; j < n; j++) {
			if (i == j) temp.push_back(1.0);
			else temp.push_back(-1.0);
		} classifiers.push_back(temp);
	}
}

OutputTarget::~OutputTarget() {

}

vector<double> OutputTarget::getOutputFromTarget(int c) {
	return classifiers[c];
}

int OutputTarget::getTargetFromOutput(vector<double> output) {
	for (int i = 0; i < classes; i++) {
		bool matches = true;
		for (int j = 0; j < nodes; j++) {
			if (abs(output[j] - classifiers[i][j]) >= 1) {
				matches = false;
				break;
			}
		}
		if (matches) return i;
	}
	return -1;
}

