/*
 * Neuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <math.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <cuda.h>
using namespace std;

class Neuron {
private:
	static long long n;
	__device__ double sigmoid(double input);
	__device__ double sigmoidPrime(double input);
	__device__ double activate(double input);
	__device__ double activatePrime(double input);
public:
	double *weightedError;
	double *weight;
	double *impulse;
	double activation;
	double activationPrime;
	int connections;
	Neuron(int nConnections);
	~Neuron();
	__device__ double forward(double *input);
	__device__ double *backward(double errorPrime, double learningRate);
	static Neuron *copyToGPU(Neuron *data);
	static Neuron *copyFromGPU(Neuron *data);
};

#endif /* NEURON_H_ */
