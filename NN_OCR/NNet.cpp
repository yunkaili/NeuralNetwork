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



float NNet::train2()
{
	srand(0);
	//	srand(0);
	// layer parameters
	int Dim0 = this->nDim;
	int Dim1 = 8;
	int Dim2 = 3;
	int Dim3 = 1;

	// training parameters
	float lr = 0.01;
	float r = 0.1;
	int iter_times = 50000;

	float *w1 = new float[(Dim0+1)*Dim1];
	for(int i = 0; i < (Dim0+1)*Dim1; i++)
		w1[i] = (float)rand() *2 * r / RAND_MAX - r ;

	float *w2 = new float[(Dim1+1) * Dim2];
	for(int i = 0; i < (Dim1+1)*Dim2; i++)
		w2[i] = (float)rand() *2 * r / RAND_MAX - r ;

	float *w3 = new float[(Dim2+1) * Dim3];
	for(int i = 0; i < (Dim2+1)*Dim3; i++)
		w3[i] = (float)rand() *2 * r / RAND_MAX - r ;

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

		// X2 = tanh(S2)
		float *x2 = new float[Dim2+1];
		x2[0] = 1;
		for(int i = 0; i < Dim2; i++)
			x2[i+1] = tanh(s2[i]);

		//	S3 = X2 * W2;
		// [Dim2 * 1] = [Dim2+1] [ (Dim2+1) * Dim3]
		float *s3 = new float[Dim3];
		float *pw3 = w3;
		for(int i = 0; i < Dim3; i++, pw3 += (Dim2+1))
		{
			float sum = 0.0f;
			for(int k = 0; k < Dim2+1; k++)
				sum += x2[k]*pw3[k];
			s3[i] = sum;
		}
		
		// Back_propagation
		// single output

		// theta3 = - 2 * (yn - s2[1])
		float theta3 = - 2 * (*py - tanh(s3[0])) / (cosh(s3[0])*cosh(s3[0]));

		float *theta2 = new float[Dim2];
		pw3 = w3+1;
		for(int i = 0; i < Dim2; i++, pw3+= Dim3)
		{
			float sum = 0.0f;
			for(int k = 0; k < Dim3; k++)
				sum += theta3 * pw3[k] / (cosh(s2[i])*cosh(s2[i]));
			theta2[i] = sum;
		}

		float *theta1 = new float[Dim1];
		pw2 = w2+1;
		for(int i = 0; i < Dim1; i++, pw2+= Dim2)
		{
			float sum = 0.0f;
			for(int k = 0; k < Dim2; k++)
				sum += theta2[i] * pw2[k] / (cosh(s1[i])*cosh(s1[i]));
			theta1[i] = sum;
		}
		
		// Update
		// w(layer)ij = w(layer)ij - lr*x(layer-1)i*theta(layer)j;

		// update w3
		pw3 = w3;
		for(int i = 0; i < Dim3;i++, pw3 += (Dim2+1))
		{
			float theta = theta3;
			for(int k = 0; k < Dim2+1; k++)
			{			
				pw3[k] -= lr*x2[k] *theta;
			}
		}

		// update w2
		pw2 = w2;
		for(int i = 0; i < Dim2;i++, pw2 += (Dim1+1))
		{
			float theta = theta2[i];
			for(int k = 0; k < Dim1+1; k++)
				pw2[k] -= lr*x1[k] * theta;
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
		delete[] x2;
		delete[] s3;
		
		delete[] theta1;
		delete[] theta2;
	}

	float error_sum = errorSum2(w1, w2, w3) / this->nDatat;
	printf("END::error_sum %f\n", error_sum	);

	delete[] w1;
	delete[] w2;
	delete[] w3;
	
	return error_sum;
}

void NNet::forward2(float *w1, float *w2, float *w3, float *xn, int yn, float &pre_lbl)
{
	int Dim0 = this->nDim;
	int Dim1 = 8;
	int Dim2 = 3;
	int Dim3 = 1;

	// Forward
	float *x0 = new float[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

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

	// X2 = tanh(S2)
	float *x2 = new float[Dim2+1];
	x2[0] = 1;
	for(int i = 0; i < Dim2; i++)
		x2[i+1] = tanh(s2[i]);

	//	S3 = X2 * W2;
	// [Dim2 * 1] = [Dim2+1] [ (Dim2+1) * Dim3]
	float *s3 = new float[Dim3];
	float *pw3 = w3;
	for(int i = 0; i < Dim3; i++, pw3 += (Dim2+1))
	{
		float sum = 0.0f;
		for(int k = 0; k < Dim2+1; k++)
			sum += x2[k]*pw3[k];
		s3[i] = sum;
	}

	pre_lbl = tanh(s3[0]) > 0 ? 1 : -1;

	delete[] x0;
	delete[] s1;
	delete[] x1;
	delete[] s2;
	delete[] x2;
	delete[] s3;
}

float NNet::errorSum2(float *w1, float *w2, float *w3)
{
	float error_sum = 0.0f;
	float *pX = this->datat;
	int *pY = this->labelt;
	for(int i = 0; i < this->nDatat; i++, pX += this->nDim, pY++)
	{
		float pre_lbl;
		forward2(w1, w2, w3, pX, *pY, pre_lbl);
		int r = *pY - pre_lbl;
		error_sum += abs(r/2);
	}
	return error_sum;
}



float NNet::gradientChecking(float *w1, float *w2, float *w3, 
							float *theta1, float *theta2, float theta3, float *xn)
{
	float epsilon = 0.0001;

	int Dim0 = this->nDim;
	int Dim1 = 8;
	int Dim2 = 3;
	int Dim3 = 1;

	float *w1t = new float[(Dim0+1)*Dim1];
	float *w2t = new float[(Dim1+1)*Dim2];
	float *w3t = new float[(Dim2+1)*Dim3];

	// check w1
	float *pw1t = w1t, *pw1 = w1;
	for(int i = 0; i < Dim0+1; i++)
	{
		for(int j = 0; j < Dim1; j++)
		{
			*pw1t = pw1 + epsilon;
			float Jplus = NNCost(w1t, w2t, w3t);
			*pw1t = pw1 - epsilon;
			float Jminus = NNCost(w1t, w2t, w3t);
			float deltaw1 = (Jplus - Jminus) / (2* epsilon);
			float deltaw0 = theta1[i] * x0[j];		
			if( abs(delta1 - delta0) > 0.001 )
				printf("gradient error!")

		}
	}



}


float NNet::NNCost(float *w1, float *w2, float *w3)
{
	float *x0 = new float[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = pdata[i];

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

	// X2 = tanh(S2)
	float *x2 = new float[Dim2+1];
	x2[0] = 1;
	for(int i = 0; i < Dim2; i++)
		x2[i+1] = tanh(s2[i]);

	//	S3 = X2 * W2;
	// [Dim2 * 1] = [Dim2+1] [ (Dim2+1) * Dim3]
	float *s3 = new float[Dim3];
	float *pw3 = w3;
	for(int i = 0; i < Dim3; i++, pw3 += (Dim2+1))
	{
		float sum = 0.0f;
		for(int k = 0; k < Dim2+1; k++)
			sum += x2[k]*pw3[k];
		s3[i] = sum;
	}

	float J = tanh(s3[0]);

	delete[] x0;
	delete[] s1;
	delete[] x1;
	delete[] s2;
	delete[] x2;
	delete[] s3;

	return J;
}