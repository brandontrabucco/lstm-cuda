/*
 * DatasetAdapter.cpp
 *
 *  Created on: Jul 26, 2016
 *      Author: trabucco
 */

#include "DatasetAdapter.h"

DatasetAdapter::DatasetAdapter() {
	// TODO Auto-generated constructor stub
	ifstream trainingDatasetFile("/stash/tlab/datasets/Language/nietzsche.txt");
	charIndex = -1;

	if (trainingDatasetFile.is_open()) {
		char buffer;
		while (trainingDatasetFile.get(buffer)) {
			dataset.text.push_back(static_cast<unsigned char>(buffer));
		} cout << "Text loaded" << endl;
		trainingDatasetFile.close();
	} else cout << "Error opening files" << endl;
}

DatasetAdapter::~DatasetAdapter() {
	// TODO Auto-generated destructor stub
}

int DatasetAdapter::getCharSize() {
	return charSize;
}

int DatasetAdapter::getDatasetSize() {
	return dataset.text.size();
}

bool DatasetAdapter::nextChar() {
	return (++charIndex < (getDatasetSize() - 1));
}

bool DatasetAdapter::isLastChar() {
	return (charIndex == (getDatasetSize() - 2));
}

DatasetExample DatasetAdapter::getChar() {
	DatasetExample example;
	example.current = dataset.text[charIndex];
	example.next = dataset.text[charIndex + 1];
	return example;
}

void DatasetAdapter::reset() {
	charIndex = -1;
}


