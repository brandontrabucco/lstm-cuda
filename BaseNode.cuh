/*
 * Neuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef BASENODE_H_
#define BASENODE_H_

#include <math.h>
#include <vector>
#include <iostream>
using namespace std;

class BaseNode {
private:
public:
	BaseNode();
	virtual ~BaseNode();
	__device__ double sigmoid(double input);
	__device__ double sigmoidPrime(double input);
	__device__ double activationFunction(double input);
	__device__ double activationFunctionPrime(double input);
};

#endif /* BASENODE_H_ */
