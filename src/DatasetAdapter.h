/*
 * DatasetAdapter.h
 *
 *  Created on: Jul 26, 2016
 *      Author: trabucco
 */

#ifndef DATASETADAPTER_H_
#define DATASETADAPTER_H_

#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

using namespace std;

typedef struct {
	vector<unsigned char> text;
} Dataset;

typedef struct {
	unsigned char current;
	unsigned char next;
} DatasetExample;

class DatasetAdapter {
private:
	Dataset dataset;
	const int charSize = 256;
	int charIndex;
public:
	DatasetAdapter();
	virtual ~DatasetAdapter();
	int getCharSize();
	int getDatasetSize();
	bool nextChar();
	bool isLastChar();
	DatasetExample getChar();
	void reset();
};

#endif /* DATASETADAPTER_H_ */
