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

	double *X = new double[num*dim];
	int *Y = new int[num];
	for(int i = 0; i < num; i++)
	{
		fread(X+i*dim, sizeof(double), dim, pf);
		fread(Y+i, sizeof(int), 1, pf);
	}

	this->nCls = cls;
	this->nDim = dim;
	this->nData = num;

	this->data = new double[num*dim];
	this->label = new int[num];

	memcpy(this->data, X, sizeof(double)*num*dim);
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

	double *X = new double[num*dim];
	int *Y = new int[num];
	for(int i = 0; i < num; i++)
	{
		fread(X+i*dim, sizeof(double), dim, pf);
		fread(Y+i, sizeof(int), 1, pf);
	}

	this->nClst = cls;
	this->nDimt = dim;
	this->nDatat = num;

	this->datat = new double[num*dim];
	this->labelt = new int[num];

	memcpy(this->datat, X, sizeof(double)*num*dim);
	memcpy(this->labelt, Y, sizeof(int)*num);

	delete[] X;
	delete[] Y;
}

double NNet::train()
{
	srand(clock());
	//	srand(0);
	// layer parameters
	int Dim0 = this->nDim;
	int Dim1 = 6;
	int Dim2 = 1;

	// training parameters
	double lr = 0.01;
	double r = 0.1;
	int iter_times = 50000;

	double *w1 = new double[(Dim0+1)*Dim1];
	for(int i = 0; i < (Dim0+1)*Dim1; i++)
		w1[i] = (double)rand() / (5*RAND_MAX)-0.1;

	double *w2 = new double[(Dim1+1) * Dim2];
	for(int i = 0; i < (Dim1+1)*Dim2; i++)
		w2[i] = (double)rand() / (5*RAND_MAX)-0.1;

	while(iter_times--)
	{
		// init
		int ridx = rand() * (this->nData-1) / RAND_MAX;
		//printf("ridx %d\n", ridx);
		double *pdata = this->data + ridx * this->nDim;
		int *py = this->label + ridx;

		// Forward
		double *x0 = new double[Dim0+1];
		x0[0] = 1;
		for(int i = 0; i < Dim0; i++)
			x0[i+1] = pdata[i];

		//double *w1 = new double[(Dim0+1)*Dim1];
		//for(int i = 0; i < (Dim0+1)*Dim1; i++)
		//	w1[i] = (double)rand() / RAND_MAX;

		//		S1 = X0 * W0;
		// [Dim1 * 1] = [Dim0+1] [Dim0+1 * Dim1]
		double *s1 = new double[Dim1];
		double *pw1 = w1;
		for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
		{
			double sum = 0.0f;
			for(int k = 0; k < Dim0+1; k++)
				sum += x0[k]*pw1[k];
			s1[i] = sum;
		}

		//	X1 = tanh(S1);
		double *x1 = new double[Dim1+1];
		x1[0] = 1;
		for(int i = 0; i<Dim1; i++)
			x1[i+1] = tanh(s1[i]);

		//double *w2 = new double[(Dim1+1) * Dim2];
		//for(int i = 0; i < (Dim1+1)*Dim2; i++)
		//	w2[i] = (double)rand() / RAND_MAX;

		//		S2 = X1 * W1;
		// [Dim2 * 1] = [Dim1+1] [(Dim1+1) * Dim2] 
		double *s2 = new double[Dim2];
		double *pw2 = w2;
		for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
		{
			double sum = 0.0f;
			for(int k = 0; k < Dim1+1; k++)
				sum += x1[k]*pw2[k];
			s2[i] = sum;
		}

		// Back_propagation
		// single output

		// theta2 = - 2 * (yn - s2[1])
		double theta2 = - 2 * (*py - tanh(s2[0])) / (cosh(s2[0])*cosh(s2[0]));

		double *theta1 = new double[Dim1];
		pw2 = w2+1;
		for(int i = 0; i < Dim1; i++, pw2+= Dim2)
		{
			double sum = 0.0f;
			for(int k = 0; k < Dim2; k++)
				sum += theta2 * pw2[k] / (cosh(s1[i])*cosh(s1[i]));
			theta1[i] = sum;
		}

		//gradientChecking(w1, w2, theta1, theta2, pdata, *py);

		// Update
		// w(layer)ij = w(layer)ij - lr*x(layer-1)i*theta(layer)j;
		
		// update w2
		pw2 = w2;
		for(int i = 0; i < Dim2;i++, pw2 += (Dim1+1))
		{
			double theta = theta2;
			for(int k = 0; k < Dim1+1; k++)
				pw2[k] -= lr*x1[k] *theta;
		}

		// update w1
		pw1 = w1;
		for(int i = 0; i < Dim1;i++, pw1 += (Dim0+1))
		{
			double theta = theta1[i];
			for(int k = 0; k < Dim0+1; k++)
				pw1[k] -= lr*x0[k] * theta;
		}

		delete[] x0;
		delete[] s1;
		delete[] x1;
		delete[] s2;
		delete[] theta1;
	}

	double error_sum = errorSum(w1, w2) / this->nDatat;
	//printf("END::error_sum %lf\n", error_sum);

	delete[] w1;
	delete[] w2;

	return error_sum;
}

void NNet::forward(double *w1, double *w2, double *xn, int yn, double &pre_lbl)
{
	int Dim0 = this->nDim;
	int Dim1 = 6;
	int Dim2 = 1;
	
	double *x0 = new double[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

	double *s1 = new double[Dim1];
	double *pw1 = w1;
	for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim0+1; k++)
			sum += x0[k]*pw1[k];
		s1[i] = sum;
	}

	double *x1 = new double[Dim1+1];
	x1[0] = 1;
	for(int i = 0; i<Dim1; i++)
		x1[i+1] = tanh(s1[i]);

	double *s2 = new double[Dim2];
	double *pw2 = w2;
	for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim1+1; k++)
			sum += x1[k]*pw2[k];
		s2[i] = sum;
	}

	pre_lbl = tanh(s2[0]) > 0 ? 1 : -1;

	delete[] s1;
	delete[] x1;
	delete[] s2;
}

double NNet::errorSum(double *w1, double *w2)
{
	double error_sum = 0.0f;
	double *pX = this->datat;
	int *pY = this->labelt;
	for(int i = 0; i < this->nDatat; i++, pX += this->nDim, pY++)
	{
		double pre_lbl;
		forward(w1, w2, pX, *pY, pre_lbl);
		int r = *pY - pre_lbl;
		error_sum += abs(r/2);
	}
	return error_sum;
}


void NNet::gradientChecking(double *w1, double *w2, double *theta1, double theta2, double *xn, int yn)
{
	double epsilon = 0.0001;

	int Dim0 = this->nDim;
	int Dim1 = 6;
	int Dim2 = 1;

	double *w1t = new double[(Dim0+1)*Dim1];
	memcpy(w1t, w1, sizeof(double)*(Dim0+1)*Dim1);
	double *w2t = new double[(Dim1+1)*Dim2];
	memcpy(w2t, w2, sizeof(double)*(Dim1+1)*Dim2);

	// check w1
	double *pw1t = w1t, *pw1 = w1;
	double *x0 = new double[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

	for(int i = 0; i < Dim1; i++)
	{
		for(int j = 0; j < Dim0+1; j++)
		{
			*pw1t = *pw1 + epsilon;
			double Jplus = NNCost(w1t, w2t, xn, yn);
			*pw1t = *pw1 - epsilon;
			double Jminus = NNCost(w1t, w2t, xn, yn);
			double deltaw1 = (Jplus - Jminus) / (2* epsilon);
			double deltaw0 = theta1[i] * x0[j];		
			if( abs(deltaw1 - deltaw0) > epsilon )
				printf("gradient error!\nformula %0.5f\tlimit %0.5f\tdiff %0.5f\n", deltaw0, deltaw1, abs(deltaw0-deltaw1));
			*pw1t = *pw1;
			pw1t++;
			pw1++;
		}
	}

	double *s1 = new double[Dim1];
	pw1 = w1;
	for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim0+1; k++)
			sum += x0[k]*pw1[k];
		s1[i] = sum;
	}
	
	// check w2
	double *pw2t = w2t, *pw2 = w2;
	double *x1 = new double[Dim1+1];
	x1[0] = 1;
	for(int i = 0; i<Dim1; i++)
		x1[i+1] = tanh(s1[i]);


	for(int i = 0; i < Dim2; i++)
	{
		for(int j = 0; j < Dim1+1; j++)
		{
			*pw2t = *pw2 + epsilon;
			double Jplus = NNCost(w1t, w2t, xn, yn);
			*pw2t = *pw2 - epsilon;
			double Jminus = NNCost(w1t, w2t, xn, yn);
			double deltaw1 = (Jplus - Jminus) / (2* epsilon);
			double deltaw0 = theta2 * x1[j];		
			if( abs(deltaw1 - deltaw0) > epsilon )
				printf("gradient error!\nformula %0.5f\tlimit %0.5f\tdiff %0.5f\n", deltaw0, deltaw1, abs(deltaw0-deltaw1));
			*pw2t = *pw2;
			pw2t++;
			pw2++;
		}
	}

	delete[] x0;
	delete[] s1;
	delete[] x1;
	delete[] w1t;
	delete[] w2t;

}


double NNet::NNCost(double *w1, double *w2, double *xn, int yn)
{
	int Dim0 = this->nDim;
	int Dim1 = 6;
	int Dim2 = 1;

	double *x0 = new double[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

	double *s1 = new double[Dim1];
	double *pw1 = w1;
	for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim0+1; k++)
			sum += x0[k]*pw1[k];
		s1[i] = sum;
	}

	//	X1 = tanh(S1);
	double *x1 = new double[Dim1+1];
	x1[0] = 1;
	for(int i = 0; i<Dim1; i++)
		x1[i+1] = tanh(s1[i]);

	//		S2 = X1 * W1;
	// [Dim2 * 1] = [Dim1+1] [(Dim1+1) * Dim2] 
	double *s2 = new double[Dim2];
	double *pw2 = w2;
	for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim1+1; k++)
			sum += x1[k]*pw2[k];
		s2[i] = sum;
	}

	double J = (yn - tanh(s2[0]));
	J *= J;

	delete[] x0;
	delete[] s1;
	delete[] x1;
	delete[] s2;

	return J;
}


double NNet::train2()
{
	srand(clock());
	//srand(0);
	
	// layer parameters
	int Dim0 = this->nDim;
	int Dim1 = 8;
	int Dim2 = 3;
	int Dim3 = 1;

	// training parameters
	double lr = 0.01;
	double r = 0.1;
	int iter_times = 50000;

	double *w1 = new double[(Dim0+1)*Dim1];
	for(int i = 0; i < (Dim0+1)*Dim1; i++)
		w1[i] = (double)rand() *2 * r / RAND_MAX - r ;

	double *w2 = new double[(Dim1+1) * Dim2];
	for(int i = 0; i < (Dim1+1)*Dim2; i++)
		w2[i] = (double)rand() *2 * r / RAND_MAX - r ;

	double *w3 = new double[(Dim2+1) * Dim3];
	for(int i = 0; i < (Dim2+1)*Dim3; i++)
		w3[i] = (double)rand() *2 * r / RAND_MAX - r ;

	while(iter_times--)
	{
		// init
		int ridx = rand() * (this->nData-1) / RAND_MAX;
		//printf("ridx %d\n", ridx);
		double *pdata = this->data + ridx * this->nDim;
		int *py = this->label + ridx;

		// Forward
		double *x0 = new double[Dim0+1];
		x0[0] = 1;
		for(int i = 0; i < Dim0; i++)
			x0[i+1] = pdata[i];

		//		S1 = X0 * W0;
		// [Dim1 * 1] = [Dim0+1] [Dim0+1 * Dim1]
		double *s1 = new double[Dim1];
		double *pw1 = w1;
		for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
		{
			double sum = 0.0f;
			for(int k = 0; k < Dim0+1; k++)
				sum += x0[k]*pw1[k];
			s1[i] = sum;
		}

		//	X1 = tanh(S1);
		double *x1 = new double[Dim1+1];
		x1[0] = 1;
		for(int i = 0; i<Dim1; i++)
			x1[i+1] = tanh(s1[i]);

		//		S2 = X1 * W1;
		// [Dim2 * 1] = [Dim1+1] [(Dim1+1) * Dim2] 
		double *s2 = new double[Dim2];
		double *pw2 = w2;
		for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
		{
			double sum = 0.0f;
			for(int k = 0; k < Dim1+1; k++)
				sum += x1[k]*pw2[k];
			s2[i] = sum;
		}

		// X2 = tanh(S2)
		double *x2 = new double[Dim2+1];
		x2[0] = 1;
		for(int i = 0; i < Dim2; i++)
			x2[i+1] = tanh(s2[i]);

		//	S3 = X2 * W2;
		// [Dim2 * 1] = [Dim2+1] [ (Dim2+1) * Dim3]
		double *s3 = new double[Dim3];
		double *pw3 = w3;
		for(int i = 0; i < Dim3; i++, pw3 += (Dim2+1))
		{
			double sum = 0.0f;
			for(int k = 0; k < Dim2+1; k++)
				sum += x2[k]*pw3[k];
			s3[i] = sum;
		}
		
		// Back_propagation
		// single output

		// theta3 = - 2 * (yn - x3[1]) * x3'[1]
		double theta3 = - 2 * (*py - tanh(s3[0])) * (1 - (tanh(s3[0]) * tanh(s3[0])));

		double *theta2 = new double[Dim2];
		for(int i = 0; i < Dim2; i++)
		{
			pw3 = w3 + i + 1;
			double sum = 0.0f;
			for(int k = 0; k < Dim3; k++, pw3 += (Dim2+1))
				sum += theta3 * (*pw3) / (cosh(s2[i])*cosh(s2[i]));
			theta2[i] = sum;
		}

		double *theta1 = new double[Dim1];
		for(int i = 0; i < Dim1; i++)
		{
			pw2 = w2 + i + 1;
			double sum = 0.0f;
			for(int k = 0; k < Dim2; k++, pw2 += (Dim1+1))
				sum += theta2[k] * (*pw2) / (cosh(s1[i])*cosh(s1[i]));
			theta1[i] = sum;
		}
		
		//gradientChecking2(w1, w2, w3, theta1, theta2, theta3, pdata, *py);

		// Update
		// w(layer)ij = w(layer)ij - lr*x(layer-1)i*theta(layer)j;

		// update w3
		pw3 = w3;
		for(int i = 0; i < Dim3;i++, pw3 += (Dim2+1))
		{
			double theta = theta3;
			for(int k = 0; k < Dim2+1; k++)
			{			
				pw3[k] -= lr*x2[k] *theta;
			}
		}

		// update w2
		pw2 = w2;
		for(int i = 0; i < Dim2;i++, pw2 += (Dim1+1))
		{
			double theta = theta2[i];
			for(int k = 0; k < Dim1+1; k++)
				pw2[k] -= lr*x1[k] * theta;
		}


		// update w1
		pw1 = w1;
		for(int i = 0; i < Dim1;i++, pw1 += (Dim0+1))
		{
			double theta = theta1[i];
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

	double error_sum = errorSum2(w1, w2, w3) / this->nDatat;
	printf("END::error_sum %lf\n", error_sum);

	delete[] w1;
	delete[] w2;
	delete[] w3;
	
	return error_sum;
}

void NNet::forward2(double *w1, double *w2, double *w3, double *xn, int yn, double &pre_lbl)
{
	int Dim0 = this->nDim;
	int Dim1 = 8;
	int Dim2 = 3;
	int Dim3 = 1;

	// Forward
	double *x0 = new double[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

	//		S1 = X0 * W0;
	// [Dim1 * 1] = [Dim0+1] [Dim0+1 * Dim1]
	double *s1 = new double[Dim1];
	double *pw1 = w1;
	for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim0+1; k++)
			sum += x0[k]*pw1[k];
		s1[i] = sum;
	}

	//	X1 = tanh(S1);
	double *x1 = new double[Dim1+1];
	x1[0] = 1;
	for(int i = 0; i<Dim1; i++)
		x1[i+1] = tanh(s1[i]);

	//		S2 = X1 * W1;
	// [Dim2 * 1] = [Dim1+1] [(Dim1+1) * Dim2] 
	double *s2 = new double[Dim2];
	double *pw2 = w2;
	for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim1+1; k++)
			sum += x1[k]*pw2[k];
		s2[i] = sum;
	}

	// X2 = tanh(S2)
	double *x2 = new double[Dim2+1];
	x2[0] = 1;
	for(int i = 0; i < Dim2; i++)
		x2[i+1] = tanh(s2[i]);

	//	S3 = X2 * W2;
	// [Dim2 * 1] = [Dim2+1] [ (Dim2+1) * Dim3]
	double *s3 = new double[Dim3];
	double *pw3 = w3;
	for(int i = 0; i < Dim3; i++, pw3 += (Dim2+1))
	{
		double sum = 0.0f;
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

double NNet::errorSum2(double *w1, double *w2, double *w3)
{
	double error_sum = 0.0f;
	double *pX = this->datat;
	int *pY = this->labelt;
	for(int i = 0; i < this->nDatat; i++, pX += this->nDimt, pY++)
	{
		double pre_lbl;
		forward2(w1, w2, w3, pX, *pY, pre_lbl);
		int r = *pY - pre_lbl;
		error_sum += abs(r/2);
	}
	return error_sum;
}



void NNet::gradientChecking2(double *w1, double *w2, double *w3, 
							double *theta1, double *theta2, double theta3, double *xn, int yn)
{
	double epsilon = 0.0001;

	int Dim0 = this->nDim;
	int Dim1 = 8;
	int Dim2 = 3;
	int Dim3 = 1;

	double *w1t = new double[(Dim0+1)*Dim1];
	memcpy(w1t, w1, sizeof(double)*(Dim0+1)*Dim1);
	double *w2t = new double[(Dim1+1)*Dim2];
	memcpy(w2t, w2, sizeof(double)*(Dim1+1)*Dim2);
	double *w3t = new double[(Dim2+1)*Dim3];
	memcpy(w3t, w3, sizeof(double)*(Dim2+1)*Dim3);

	double *x0 = new double[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

	// check w1
	double *pw1t = w1t, *pw1 = w1;
	for(int i = 0; i < Dim1; i++)
	{
		for(int j = 0; j < Dim0+1; j++)
		{
			*pw1t = *pw1 + epsilon;
			double Jplus = NNCost2(w1t, w2t, w3t, xn, yn);
			*pw1t = *pw1 - epsilon;
			double Jminus = NNCost2(w1t, w2t, w3t, xn, yn);
			double deltaw1 = (Jplus - Jminus) / (2* epsilon);
			double deltaw0 = theta1[i] * x0[j];		
			if( abs(deltaw1 - deltaw0) > epsilon )
				printf("w1 gradient error!\nformula %0.5f\tlimit %0.5f\tdiff %0.5f\n", deltaw0, deltaw1, abs(deltaw0-deltaw1));
			*pw1t = *pw1;
			pw1t++;
			pw1++;

		}
	}

	double *s1 = new double[Dim1];
	pw1 = w1;
	for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim0+1; k++)
			sum += x0[k]*pw1[k];
		s1[i] = sum;
	}

	// check w2
	double *pw2t = w2t, *pw2 = w2;
	double *x1 = new double[Dim1+1];
	x1[0] = 1;
	for(int i = 0; i<Dim1; i++)
		x1[i+1] = tanh(s1[i]);


	for(int i = 0; i < Dim2; i++)
	{
		for(int j = 0; j < Dim1+1; j++)
		{
			*pw2t = *pw2 + epsilon;
			double Jplus = NNCost2(w1t, w2t, w3t, xn, yn);
			*pw2t = *pw2 - epsilon;
			double Jminus = NNCost2(w1t, w2t, w3t, xn, yn);
			double deltaw1 = (Jplus - Jminus) / (2* epsilon);
			double deltaw0 = theta2[i] * x1[j];		
			if( abs(deltaw1 - deltaw0) > epsilon )
				printf("w2 gradient error!\nformula %0.5f\tlimit %0.5f\tdiff %0.5f\n", deltaw0, deltaw1, abs(deltaw0-deltaw1));
			*pw2t = *pw2;
			pw2t++;
			pw2++;
		}
	}

	double *s2 = new double[Dim2];
	pw2 = w2;
	for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim1+1; k++)
			sum += x1[k]*pw2[k];
		s2[i] = sum;
	}

	double *x2 = new double[Dim2+1];
	x2[0] = 1;
	for(int i = 0; i < Dim2; i++)
		x2[i+1] = tanh(s2[i]);

	// check w3
	double *pw3t = w3t, *pw3 = w3;
	for(int i = 0; i < Dim3; i++)
	{
		for(int j = 0; j < Dim2+1; j++)
		{
			*pw3t = *pw3 + epsilon;
			double Jplus = NNCost2(w1t, w2t, w3t, xn, yn);
			*pw3t = *pw3 - epsilon;
			double Jminus = NNCost2(w1t, w2t, w3t, xn, yn);
			double deltaw1 = (Jplus - Jminus) / (2*epsilon);	
			double deltaw0 = theta3 * x2[j];
			if(abs(deltaw0 - deltaw1) > epsilon)
				printf("w3 gradient error!\nformula %0.5f limit %0.5f diff %0.5f\n", deltaw0, deltaw1, abs(deltaw0-deltaw1));
			*pw3t = *pw3;
			pw3t++;
			pw3++;
		}

	}

	delete[] w1t;
	delete[] w2t;
	delete[] w3t;

	delete[] x0;
	delete[] s1;
	delete[] x1;
	delete[] s2;
	delete[] x2;

}


double NNet::NNCost2(double *w1, double *w2, double *w3, double *xn, int yn)
{
	int Dim0 = this->nDim;
	int Dim1 = 8;
	int Dim2 = 3;
	int Dim3 = 1;

	double *x0 = new double[Dim0+1];
	x0[0] = 1;
	for(int i = 0; i < Dim0; i++)
		x0[i+1] = xn[i];

	double *s1 = new double[Dim1];
	double *pw1 = w1;
	for(int i = 0; i < Dim1; i++, pw1+= (Dim0+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim0+1; k++)
			sum += x0[k]*pw1[k];
		s1[i] = sum;
	}

	//	X1 = tanh(S1);
	double *x1 = new double[Dim1+1];
	x1[0] = 1;
	for(int i = 0; i<Dim1; i++)
		x1[i+1] = tanh(s1[i]);

	//		S2 = X1 * W1;
	// [Dim2 * 1] = [Dim1+1] [(Dim1+1) * Dim2] 
	double *s2 = new double[Dim2];
	double *pw2 = w2;
	for(int i = 0; i < Dim2; i++, pw2+= (Dim1+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim1+1; k++)
			sum += x1[k]*pw2[k];
		s2[i] = sum;
	}

	// X2 = tanh(S2)
	double *x2 = new double[Dim2+1];
	x2[0] = 1;
	for(int i = 0; i < Dim2; i++)
		x2[i+1] = tanh(s2[i]);

	//	S3 = X2 * W2;
	// [Dim2 * 1] = [Dim2+1] [ (Dim2+1) * Dim3]
	double *s3 = new double[Dim3];
	double *pw3 = w3;
	for(int i = 0; i < Dim3; i++, pw3 += (Dim2+1))
	{
		double sum = 0.0f;
		for(int k = 0; k < Dim2+1; k++)
			sum += x2[k]*pw3[k];
		s3[i] = sum;
	}

	double J = yn - tanh(s3[0]);
	J *= J;

	delete[] x0;
	delete[] s1;
	delete[] x1;
	delete[] s2;
	delete[] x2;
	delete[] s3;

	return J;
}