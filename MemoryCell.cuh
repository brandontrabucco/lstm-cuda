/*
 * MemoryCell.h
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#ifndef MEMORYCELL_H_
#define MEMORYCELL_H_

#include "BaseNode.cuh"
#include <vector>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <random>
using namespace std;

class MemoryCell : BaseNode {
private:
	static long long n;
public:
	int nConnections;
	double *cellDataWeight, *cellDataPartial,
		*inputDataPartial, *forgetDataPartial;
	double cellFeedbackWeight, bias;
	double activationIn, activationInPrime,
		activationOut, activationOutPrime,
		state, previousState,
		feedback, previousFeedback,
		cellFeedbackPartial;
	double inputFeedbackPartial, inputStatePartial,
		forgetFeedbackPartial, forgetStatePartial;
	__device__ double activateIn(double data);
	__device__ double activateOut(double data);
	MemoryCell(int c);
	virtual ~MemoryCell();
	static MemoryCell *copyToGPU(MemoryCell *memory);
	static MemoryCell *copyFromGPU(MemoryCell *memory);
};

#endif /* MEMORYCELL_H_ */
