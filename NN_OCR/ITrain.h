#ifndef _ITRAIN_HEAD_H
#define _ITRAIN_HEAD_H

#include "IMatrix.h"
#include "NNLayer.h"
#include <vector>

class ITrain
{
public:
	ITrain(void);
	~ITrain(void);

	void ImportTrainingData(char *dataFile);
	void ImportTestingData(char *dataFile);

	void config();
	void train();
	void test_Ein();
	void test_Eout();

	void clear_config();

	std::vector<NNLayer*> nnet;

private:

	// Training Data
	double *data;
	int *label;
	int nData;
	int nCls;
	int nDim;

	// Testing Data
	double *datat;
	int *labelt;
	int nDatat;
	int nClst;
	int nDimt;

};

#endif