/*
 * MemoryCell.cpp
 *
 *  Created on: Jul 18, 2016
 *      Author: trabucco
 */

#include "MemoryCell.cuh"

long long MemoryCell::n = 0;

MemoryCell::MemoryCell(int c) {
	// TODO Auto-generated constructor stub
	nConnections = c;
	activationIn = 0; activationInPrime = 0;
	activationOut = 0; activationOutPrime = 0;
	state = 0; previousState = 0;
	feedback = 0; previousFeedback = 0;
	bias = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	cellFeedbackWeight = d(g);
	cellFeedbackPartial = 0;
	inputFeedbackPartial = 0;
	inputStatePartial = 0;
	forgetFeedbackPartial = 0;
	forgetStatePartial = 0;

	cellDataWeight = (double *)malloc(sizeof(double) * c);
	cellDataPartial = (double *)malloc(sizeof(double) * c);
	forgetDataPartial = (double *)malloc(sizeof(double) * c);
	inputDataPartial = (double *)malloc(sizeof(double) * c);

	for (int i = 0; i < c; i++) {
		cellDataWeight[i] = (d(g));
		cellDataPartial[i] = (0);
		forgetDataPartial[i] = (0);
		inputDataPartial[i] = (0);
	}
}

MemoryCell::~MemoryCell() {
	// TODO Auto-generated destructor stub
}

__device__ double MemoryCell::activateIn(double data) {
	activationIn = activationFunction(data);
	activationInPrime = activationFunctionPrime(data);
	return activationIn;
}

__device__ double MemoryCell::activateOut(double data) {
	activationOut = activationFunction(data);
	activationOutPrime = activationFunctionPrime(data);
	return activationOut;
}

MemoryCell *MemoryCell::copyToGPU(MemoryCell *memory) {
	MemoryCell *memoryCell;
	//cout << "Test " << memory << endl;
	cudaMalloc((void **)&memoryCell, (sizeof(MemoryCell)));
	cudaDeviceSynchronize();
	//cout << "Test" << endl;
	cudaMemcpy(memoryCell, memory, sizeof(MemoryCell), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	//cout << "Test" << endl;

	double *cdw, *idp, *fdp, *cdp;
	cudaMalloc((void **)&cdw, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&idp, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&fdp, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&cdp, (sizeof(double) * memory->nConnections));
	cudaDeviceSynchronize();

	cudaMemcpy(cdw, memory->cellDataWeight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(idp, memory->inputDataPartial, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(fdp, memory->forgetDataPartial, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(cdp, memory->cellDataPartial, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(&(memoryCell->cellDataWeight), &cdw, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryCell->inputDataPartial), &idp, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryCell->forgetDataPartial), &fdp, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryCell->cellDataPartial), &cdp, sizeof(double *), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	return memoryCell;
}

MemoryCell *MemoryCell::copyFromGPU(MemoryCell *memory) {

	MemoryCell *memoryCell;
	memoryCell = (MemoryCell *)malloc((sizeof(MemoryCell)));
	cudaDeviceSynchronize();
	cudaMemcpy(memoryCell, memory, sizeof(MemoryCell), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double *cdw, *idp, *fdp, *cdp;
	cdw = (double *)malloc(sizeof(double) * memoryCell->nConnections);
	idp = (double *)malloc(sizeof(double) * memoryCell->nConnections);
	fdp = (double *)malloc(sizeof(double) * memoryCell->nConnections);
	cdp = (double *)malloc(sizeof(double) * memoryCell->nConnections);

	cudaMemcpy(cdw, memoryCell->cellDataWeight, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(idp, memoryCell->inputDataPartial, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(fdp, memoryCell->forgetDataPartial, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(cdp, memoryCell->cellDataPartial, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	memcpy(&(memoryCell->cellDataWeight), &cdw, (sizeof(double *)));
	memcpy(&(memoryCell->inputDataPartial), &idp, (sizeof(double *)));
	memcpy(&(memoryCell->forgetDataPartial), &fdp, (sizeof(double *)));
	memcpy(&(memoryCell->cellDataPartial), &cdp, (sizeof(double *)));

	return memoryCell;
}

