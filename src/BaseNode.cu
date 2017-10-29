/*
 * Neuron.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "BaseNode.cuh"

BaseNode::BaseNode() {
	// TODO Auto-generated constructor stub
}

BaseNode::~BaseNode() {
	// TODO Auto-generated destructor stub
}

__device__ double BaseNode::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

__device__ double BaseNode::sigmoidPrime(double input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

__device__ double BaseNode::activationFunction(double input) {
	return tanh(input);
}

__device__ double BaseNode::activationFunctionPrime(double input) {
	return (1 - (tanh(input) * tanh(input)));
}

