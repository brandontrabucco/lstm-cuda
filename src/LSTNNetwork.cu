/*
 * LSTMNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "LSTMNetwork.cuh"

__global__ void forwardPass(Neuron **neurons, double *connections, double *activations, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			activations[idx] = neurons[idx]->forward(connections);
			//printf("F Neuron %d : %f\n", size, activations[idx]);
		}
	}
}

__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum, double learningRate, int connections, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			double *contribution = neurons[idx]->backward(weightedError[idx], learningRate);
			//printf("B Neurons %d\n", size);
			for (int j = 0; j < connections; j++) {
				errorSum[j] += contribution[j];
			}
		}
	}
}

__global__ void forwardPassLSTM(MemoryBlock **blocks, double *connections, double *activations, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			double *blockActivation = blocks[idx]->forward(connections);
			//printf("F Cells %d\n", blocks[idx]->nCells);
			for (int j = 0; j < blocks[i]->nCells; j++) activations[idx * blocks[i]->nCells + j] = blockActivation[j];
		}
	}
}

__global__ void backwardPassLSTM(MemoryBlock **blocks, double **weightedError, double *errorSum, double learningRate, int connections, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			if (idx == 0) printf("B Cells %p\n", blocks[idx]);
			double *contribution = blocks[idx]->backward(weightedError[idx], learningRate);
			for (int j = 0; j < connections; j++) {
				errorSum[j] += contribution[j];
			}
		}
	}
}

LSTMNetwork::LSTMNetwork(int is, int b, int c, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	learningRate = l;
	decayRate = d;
	for (int i = 0; i < b; i++) {
		blocks.push_back(MemoryBlock(c, is));
	}
}

LSTMNetwork::~LSTMNetwork() {
	// TODO Auto-generated destructor stub
}

int LSTMNetwork::getPreviousNeurons() {
	return (layers.size() == 0) ? (blocks.size() * blocks[0].nCells) : layers[layers.size() - 1].size();
}

void LSTMNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} layers.push_back(buffer);
}

vector<double> LSTMNetwork::classify(vector<double> input) {
	double *output = (double *)malloc(sizeof(double) * blocks.size() * blocks[0].nCells),
			*connections;
	cudaMalloc((void **)&connections, sizeof(double) * input.size());
	cudaMemcpy(&connections[0], &input[0], (sizeof(double) * input.size()), cudaMemcpyHostToDevice);
	if (input.size() == inputSize) {
		// calculate activations from bottom up
		double *activations;
		cudaMalloc((void **)&activations, (sizeof(double) * blocks.size() * blocks[0].nCells));

		MemoryBlock **deviceBlocks, **blockBuffer = (MemoryBlock **)malloc(sizeof(MemoryBlock *) * blocks.size());
		for (int i = 0; i < blocks.size(); i++) {
			cudaMemcpy(&(blocks[i].impulse[0]), &connections[0], (sizeof(double) * blocks[i].nConnections), cudaMemcpyDeviceToHost);
		}
		cudaMalloc((void **)&deviceBlocks, sizeof(MemoryBlock *) * blocks.size());
		for (int i = 0; i < blocks.size(); i++) {
			MemoryBlock *db = MemoryBlock::copyToGPU(&blocks[i]);
			cudaMemcpy(&deviceBlocks[i], &db, sizeof(MemoryBlock *), cudaMemcpyHostToDevice);
		} forwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, connections, activations, blocks.size(), ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
		cudaDeviceSynchronize();

		cudaMemcpy(&blockBuffer[0], &deviceBlocks[0], (sizeof(MemoryBlock *) * blocks.size()), cudaMemcpyDeviceToHost);
		for (int i = 0; i < blocks.size(); i++) {
			blocks[i] = *MemoryBlock::copyFromGPU(blockBuffer[i]);
		} free(blockBuffer);
		cudaFree(deviceBlocks);

		cudaFree(connections);
		cudaMalloc((void **)&connections, (sizeof(double) * blocks.size() * blocks[0].nCells));
		cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * blocks.size() * blocks[0].nCells), cudaMemcpyDeviceToDevice);
		cudaFree(activations);
		free(output);
		output = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size());

		for (int i = 0; i < layers.size(); i++) {
			cudaMalloc((void **)&activations, (sizeof(double) * layers[i].size()));

			Neuron **deviceNeurons, **neuronBuffer = (Neuron **)malloc(sizeof(Neuron *) * layers[i].size());
			for (int j = 0; j < layers[i].size(); j++) {
				cudaMemcpy(&(layers[i][j].impulse[0]), &connections[0], (sizeof(double) * layers[i][j].connections), cudaMemcpyDeviceToHost);
			}
			cudaMalloc((void **)&deviceNeurons, sizeof(Neuron *) * layers[i].size());
			for (int j = 0; j < layers[i].size(); j++) {
				Neuron *dn = Neuron::copyToGPU(&layers[i][j]);
				cudaMemcpy(&deviceNeurons[j], &dn, sizeof(Neuron *), cudaMemcpyHostToDevice);
			} forwardPass<<<maxBlocks, maxThreads>>>(deviceNeurons, connections, activations, layers[i].size(), ceil((double)layers[i].size() / (double)(maxBlocks * maxThreads)));
			cudaDeviceSynchronize();

			cudaFree(connections);
			cudaMalloc((void **)&connections, (sizeof(double) * layers[i].size()));
			cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * layers[i].size()), cudaMemcpyDeviceToDevice);
			cudaMemcpy(&neuronBuffer[0], &deviceNeurons[0], (sizeof(Neuron *) * layers[i].size()), cudaMemcpyDeviceToHost);
			for (int j = 0; j < layers[i].size(); j++) {
				layers[i][j] = *Neuron::copyFromGPU(neuronBuffer[j]);
			} if (i == (layers.size() - 1)) cudaMemcpy(&output[0], &activations[0], (sizeof(double) * layers[layers.size() - 1].size()), cudaMemcpyDeviceToHost);
			cudaFree(activations);
			cudaFree(deviceNeurons);
			free(neuronBuffer);
		} vector<double> result(&output[0], &output[layers[layers.size() - 1].size()]);
		free(output);
		cudaFree(connections);
		return result;
	} else return vector<double>();
}

vector<double> LSTMNetwork::train(vector<double> input, vector<double> target) {
	Neuron ***deviceNeurons = (Neuron ***)malloc(sizeof(Neuron *) * layers.size());
	double *output = (double *)malloc(blocks.size() * blocks[0].nCells * sizeof(double)),
			*connections;
	cudaMalloc((void **)&connections, sizeof(double) * input.size());
	cudaMemcpy(&connections[0], &input[0], (sizeof(double) * input.size()), cudaMemcpyHostToDevice);
	if (input.size() == inputSize) {
		// start forward pass
		// calculate activations from bottom up
		double *activations;
		cudaMalloc((void **)&activations, (sizeof(double) * blocks.size() * blocks[0].nCells));
		MemoryBlock **deviceBlocks;
		for (int i = 0; i < blocks.size(); i++) {
			cudaMemcpy(&(blocks[i].impulse[0]), &connections[0], (sizeof(double) * blocks[i].nConnections), cudaMemcpyDeviceToHost);
		} cudaMalloc((void **)&deviceBlocks, sizeof(MemoryBlock *) * blocks.size());
		for (int i = 0; i < blocks.size(); i++) {
			cout << "Test " << blocks[i].cells[0]->nConnections << endl;
			MemoryBlock *db = MemoryBlock::copyToGPU(&blocks[i]);
			cudaMemcpy(&deviceBlocks[i], &db, sizeof(MemoryBlock *), cudaMemcpyHostToDevice);
		} forwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, connections, activations, blocks.size(), ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
		cudaDeviceSynchronize();
		cudaFree(connections);
		cudaMalloc((void **)&connections, (sizeof(double) * blocks.size() * blocks[0].nCells));
		cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * blocks.size() * blocks[0].nCells), cudaMemcpyDeviceToDevice);
		cudaFree(activations);
		free(output);
		output = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size());

		for (int i = 0; i < layers.size(); i++) {
			cudaMalloc((void **)&activations, (sizeof(double) * layers[i].size()));

			Neuron **layerNeurons;
			for (int j = 0; j < layers[i].size(); j++) {
				cudaMemcpy(&(layers[i][j].impulse[0]), &connections[0], (sizeof(double) * layers[i][j].connections), cudaMemcpyDeviceToHost);
			}
			cudaMalloc((void **)&layerNeurons, sizeof(Neuron *) * layers[i].size());
			for (int j = 0; j < layers[i].size(); j++) {
				Neuron *dn = Neuron::copyToGPU(&layers[i][j]);
				cudaMemcpy(&layerNeurons[j], &dn, sizeof(Neuron *), cudaMemcpyHostToDevice);
			} deviceNeurons[i] = layerNeurons;
			forwardPass<<<maxBlocks, maxThreads>>>(layerNeurons, connections, activations, layers[i].size(), ceil((double)layers[i].size() / (double)(maxBlocks * maxThreads)));
			cudaDeviceSynchronize();
			cudaFree(connections);
			cudaMalloc((void **)&connections, (sizeof(double) * layers[i].size()));
			cout << "copy activations " << cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * layers[i].size()), cudaMemcpyDeviceToDevice);
			cudaFree(activations);
		} cudaFree(connections);

		// start backward pass
		double *weightedError;
		cudaMalloc((void **)&weightedError, (sizeof(double) * layers[layers.size() - 1].size()));
		for (int i = 0; i < layers[layers.size() - 1].size(); i++) {
			double error = (output[i] - target[i]);
			output[i] = error;
			cudaMemcpy(&weightedError[i], &error, sizeof(double), cudaMemcpyHostToDevice);
		} for (int i = (layers.size() - 1); i >= 0; i--) {
			double *errorSum;
			cudaMalloc((void **)&errorSum, (sizeof(double) * layers[i][0].connections));
			cudaMemset(&errorSum[0], 0, (sizeof(double) * layers[i][0].connections));

			// compute the gradient
			backwardPass<<<maxBlocks, maxThreads>>>(deviceNeurons[i], weightedError, errorSum, learningRate, layers[i][0].connections, layers[i].size(), ceil((double)layers[i].size() / (double)(maxBlocks * maxThreads)));
			cudaDeviceSynchronize();
			cudaFree(weightedError);
			cudaMalloc((void **)&weightedError, (sizeof(double) * layers[i][0].connections));
			cout << "copy sum " << cudaMemcpy(&weightedError[0], &errorSum[0], (sizeof(double) * layers[i][0].connections), cudaMemcpyDeviceToDevice);

			Neuron **neuronBuffer = (Neuron **)malloc(sizeof(Neuron) * layers[i].size());
			cout << "copy neurons " << cudaMemcpy(&neuronBuffer[0], &deviceNeurons[i][0], (sizeof(Neuron *) * layers[i].size()), cudaMemcpyDeviceToHost);
			for (int j = 0; j < layers[i].size(); j++) {
				layers[i][j] = *Neuron::copyFromGPU(neuronBuffer[j]);
			} free(neuronBuffer);
			cudaFree(deviceNeurons[i]);
		}
		double **errorChunks, *errorSum;
		cudaMalloc((void **)&errorChunks, (sizeof(double *) * blocks.size()));
		cudaMalloc((void **)&errorSum, (sizeof(double) * blocks[0].nConnections));
		cudaMemset(&errorSum[0], 0.0, (sizeof(double) * blocks[0].nConnections));
		for (int i = 0; i < (blocks.size()); i++) {
			double *chunk;
			cudaMalloc((void **)&chunk, (sizeof(double) * blocks[i].nCells));
			cudaMemcpy(&chunk[0], &weightedError[(i * blocks[i].nCells)], (sizeof(double) * blocks[i].nCells), cudaMemcpyDeviceToDevice);
			cudaMemcpy(&errorChunks[i], &chunk, (sizeof(double *)), cudaMemcpyHostToDevice);
		} backwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, errorChunks, errorSum, learningRate, blocks[0].nConnections, blocks.size(), ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
		cudaDeviceSynchronize();

		MemoryBlock **blockBuffer = (MemoryBlock **)malloc(sizeof(MemoryBlock *) * blocks.size());
		cout << blocks.size() << " copy blocks " << cudaMemcpy(blockBuffer, deviceBlocks, (sizeof(MemoryBlock *) * blocks.size()), cudaMemcpyDeviceToHost);

		cout << "CB  " << blockBuffer[0] << endl;

		for (int i = 0; i < blocks.size(); i++) {
			MemoryBlock temp = *MemoryBlock::copyFromGPU(blockBuffer[i]);
			blocks[i] = temp;
			cout << "Test copy " << blocks[i].cells[0]->nConnections << endl;
		}

		cudaFree(deviceBlocks);
		free(deviceNeurons);
		cudaFree(weightedError);
		cudaFree(errorChunks);
		cudaFree(errorSum);

		learningRate *= decayRate;
		vector<double> result(&output[0], &output[layers[layers.size() - 1].size()]);
		free(output);
		return result;
	} else {
		cout << "Target size mismatch" << endl;
		return vector<double>();
	}
}
