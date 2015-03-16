#include "NNet.h"
#include <stdio.h>
#include <string>
#include <math.h>
#include <time.h>

#define NUMERICAL_CHECK 0

NNet::NNet(void)
{
	this->data = NULL;
	this->label = NULL;
	this->datat = NULL;
	this->labelt = NULL;
}

NNet::~NNet(void)
{
	if(!this->data) delete[] this->data;
	if(!this->label)delete[] this->label;

	if(!this->datat) delete[] this->datat;
	if(!this->labelt)delete[] this->labelt;
}

void NNet::ImportTrainingData(char *dataFile)
{
	FILE *pf;
	pf = fopen(dataFile, "rb");
	int cls;
	int dim;
	int num;
	fread(&cls, sizeof(int), 1, pf);
	fread(&dim, sizeof(int), 1, pf);
	fread(&num, sizeof(int), 1, pf);

	float *X = new float[num*dim];
	int *Y = new int[num];
	for(int i = 0; i < num; i++)
	{
		fread(X+i*dim, sizeof(float), dim, pf);
		fread(Y+i, sizeof(int), 1, pf);
	}

	this->nCls = cls;
	this->nDim = dim;
	this->nData = num;

	this->data = new float[num*dim];
	this->label = new int[num];

	memcpy(this->data, X, sizeof(float)*num*dim);
	memcpy(this->label, Y, sizeof(int)*num);

	delete[] X;
	delete[] Y;
}

void NNet::ImportTestingData(char *dataFile)
{
	FILE *pf;
	pf = fopen(dataFile, "rb");
	int cls;
	int dim;
	int num;
	fread(&cls, sizeof(int), 1, pf);
	fread(&dim, sizeof(int), 1, pf);
	fread(&num, sizeof(int), 1, pf);

	float *X = new float[num*dim];
	int *Y = new int[num];
	for(int i = 0; i < num; i++)
	{
		fread(X+i*dim, sizeof(float), dim, pf);
		fread(Y+i, sizeof(int), 1, pf);
	}

	this->nClst = cls;
	this->nDimt = dim;
	this->nDatat = num;

	this->datat = new float[num*dim];
	this->labelt = new int[num];

	memcpy(this->datat, X, sizeof(float)*num*dim);
	memcpy(this->labelt, Y, sizeof(int)*num);

	delete[] X;
	delete[] Y;
}

float NNet::train()
{
	srand(clock());
	//	srand(0);
	// layer parameters
	int Dim0 = this->nDim;
	int Dim1 = 6;
	int Dim2 = 1;

	// training parameters
	float lr = 0.1;
	float r = 0.1;
	int iter_times = 50000;

	float *w1 = new float[(Dim0+1)*Dim1];
	for(int i = 0; i < (Dim0+1)*Dim1; i++)
		w1[i] = (float)rand() / (5*RAND_MAX)-0.1;

	float *w2 = new float[(Dim1+1) * Dim2];
	for(int i = 0; i < (Dim1+1)*Dim2; i++)
		w2[i] = (float)rand() / (5*RAND_MAX)-0.1;

	while(iter_times--)
	{
		// init
		int ridx = rand() * (this->nData-1) / RAND_MAX;
		//printf("ridx %d\n", ridx);
		float *pdata = this->data + ridx * this->nDim;
		int *py = this->label + ridx;

		// Forward
		float *x0 = new float[Dim0+1];
		x0[0] = 1;
		for(int i = 0; i < Dim0; i++)
			x0[i+1] = pdata[i];

		//float *w1 = new float[(Dim0+1)*Dim1];
		//for(int i = 0; i < (Dim0+1)*Dim1; i++)
		//	w1[i] = (float)rand() / RAND_MAX;

		//		S1 = X0 * W0;
		// [Dim1 * 1] = [Dim0+1] [Dim0+1 * Dim1]
		float *s1 = new float[Dim1];
		float *pw1 = w1;
		for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
		{
			float sum = 0.0f;
			for(int k = 0; k < Dim0+1; k++)
				sum += x0[k]*pw1[k];
			s1[i] = sum;
		}

		//	X1 = tanh(S1);
		float *x1 = new float[Dim1+1];
		x1[0] = 1;
		for(int i = 0; i<Dim1; i++)
			x1[i+1] = tanh(s1[i]);

		//float *w2 = new float[(Dim1+1) * Dim2];
		//for(int i = 0; i < (Dim1+1)*Dim2; i++)
		//	w2[i] = (float)rand() / RAND_MAX;

		//		S2 = X1 * W1;
		// [Dim2 * 1] = [Dim1+1] [(Dim1+1) * Dim2] 
		float *s2 = new float[Dim2];
		float *pw2 = w2;
		for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
		{
			float sum = 0.0f;
			for(int k = 0; k < Dim1+1; k++)
				sum += x1[k]*pw2[k];
			s2[i] = sum;
		}

		// Back_propagation
		// single output

		// theta2 = - 2 * (yn - s2[1])
		float theta2 = - 2 * (*py - tanh(s2[0])) / (cosh(s2[0])*cosh(s2[0]));

		float *theta1 = new float[Dim1];
		pw2 = w2+1;
		for(int i = 0; i < Dim1; i++, pw2+= Dim2)
		{
			float sum = 0.0f;
			for(int k = 0; k < Dim2; k++)
				sum += theta2 * pw2[k] / (cosh(s1[i])*cosh(s1[i]));
			theta1[i] = sum;
		}

		// Update
		// w(layer)ij = w(layer)ij - lr*x(layer-1)i*theta(layer)j;
		
		// update w2
		pw2 = w2;
		for(int i = 0; i < Dim2;i++, pw2 += (Dim1+1))
		{
			float theta = theta2;

#if NUMERICAL_CHECK
			ptrCheck(w2, pw2, 1000.0f, 7);
#endif

			for(int k = 0; k < Dim1+1; k++)
			{			
				pw2[k] -= lr*x1[k] *theta;
#if NUMERICAL_CHECK
				ptrCheck(w2, pw2, 1000.0f, 7);
#endif

			}
		}

		// update w1
		pw1 = w1;
		for(int i = 0; i < Dim1;i++, pw1 += (Dim0+1))
		{
			float theta = theta1[i];
			for(int k = 0; k < Dim0+1; k++)
				pw1[k] -= lr*x0[k] * theta;
		}

		delete[] x0;
		delete[] s1;
		delete[] x1;
		delete[] s2;
		delete[] theta1;
	}

	float error_sum = errorSum(w1, w2) / this->nDatat;
	printf("END::error_sum %f\n", error_sum	);

	delete[] w1;
	delete[] w2;

	return error_sum;
}

void NNet::forward(float *w1, float *w2, float *xn, int yn, float &pre_lbl)
{
	int Dim0 = this->nDim;
	int Dim1 = 6;
	int Dim2 = 1;
	
	float *x0 = new float[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

	float *s1 = new float[Dim1];
	float *pw1 = w1;
	for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
	{
		float sum = 0.0f;
		for(int k = 0; k < Dim0+1; k++)
			sum += x0[k]*pw1[k];
		s1[i] = sum;
	}

	float *x1 = new float[Dim1+1];
	x1[0] = 1;
	for(int i = 0; i<Dim1; i++)
		x1[i+1] = tanh(s1[i]);

	float *s2 = new float[Dim2];
	float *pw2 = w2;
	for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
	{
		float sum = 0.0f;
		for(int k = 0; k < Dim1+1; k++)
			sum += x1[k]*pw2[k];
		s2[i] = sum;
	}

	pre_lbl = tanh(s2[0]) > 0 ? 1 : -1;

	delete[] s1;
	delete[] x1;
	delete[] s2;
}

float NNet::errorSum(float *w1, float *w2)
{
	float error_sum = 0.0f;
	float *pX = this->datat;
	int *pY = this->labelt;
	for(int i = 0; i < this->nDatat; i++, pX += this->nDim, pY++)
	{
		float pre_lbl;
		forward(w1, w2, pX, *pY, pre_lbl);
		int r = *pY - pre_lbl;
		error_sum += abs(r/2);
	}
	return error_sum;
}
