/*
 * OutputTarget.h
 *
 *  Created on: Jun 23, 2016
 *      Author: trabucco
 */

#ifndef OUTPUTTARGET_H_
#define OUTPUTTARGET_H_

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class OutputTarget {
private:
	int nodes = 256;
	int classes = 256;
	vector<vector<double> > classifiers;
public:
	OutputTarget(int n, int c);
	~OutputTarget();
	vector<double> getOutputFromTarget(int c);
	int getTargetFromOutput(vector<double> output);
};

#endif /* OUTPUTTARGET_H_ */
