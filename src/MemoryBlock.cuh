/*
 * MemoryBlock.h
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#ifndef MEMORYBLOCK_H_
#define MEMORYBLOCK_H_

#include "BaseNode.cuh"
#include "MemoryCell.cuh"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <random>
using namespace std;

class MemoryBlock : public BaseNode {
private:
	static long long n;
public:
	int nConnections;
	int nCells;
	MemoryCell **cells;
	double *inputDataWeight,
		*forgetDataWeight, *outputDataWeight,
		*bias, *impulse,
		*inputFeedbackWeight, *inputStateWeight,
		*forgetFeedbackWeight, *forgetStateWeight,
		*outputFeedbackWeight, *outputStateWeight;
	double input, inputPrime,
		forget, forgetPrime,
		output, outputPrime;
	__device__ double inputGate(double data);
	__device__ double forgetGate(double data);
	__device__ double outputGate(double data);
	MemoryBlock(int cl, int cn);
	virtual ~MemoryBlock();
	__device__ double *forward(double *input);
	__device__ double *backward(double *errorPrime, double learningRate);
	static MemoryBlock *copyToGPU(MemoryBlock *memory);
	static MemoryBlock *copyFromGPU(MemoryBlock *memory);
};

#endif /* MEMORYBLOCK_H_ */
