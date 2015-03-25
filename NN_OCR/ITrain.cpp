#include "ITrain.h"
#include <time.h>

ITrain::ITrain(void)
{
	this->data = NULL;
	this->label = NULL;
	this->datat = NULL;
	this->labelt = NULL;
}

ITrain::~ITrain(void)
{
	if(!this->data) delete[] this->data;
	if(!this->label)delete[] this->label;
	if(!this->datat) delete[] this->datat;
	if(!this->labelt)delete[] this->labelt;

	int size = nnet.size();
	while(size)
	{
		NNLayer *tmp = nnet.back();
		nnet.pop_back();
		delete[] tmp;
		size--;
	}
	nnet.clear();
	
}

void ITrain::ImportTrainingData(char *dataFile)
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
	fclose(pf);

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

void ITrain::ImportTestingData(char *dataFile)
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
	fclose(pf);
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

void ITrain::clear_config()
{
	int size = nnet.size();
	while(size)
	{
		NNLayer *tmp = nnet.back();
		nnet.pop_back();
		delete tmp;
		size--;
	}
	nnet.clear();
}

void ITrain::config()
{
	double r = 0.1;
	double lr = 0.01;
	srand(clock());	
	NNLayer *input_layer = new NNLayer(3, 9, r, lr, "input_layer");
	NNLayer *hidden_layer1 = new NNLayer(9, 4, r, lr, "hidden_layer1");
	NNLayer *output_layer = new NNLayer(4, 2, r, lr, "output_layer");

	nnet.push_back(input_layer);
	nnet.push_back(hidden_layer1);
	nnet.push_back(output_layer);

	for(std::vector<NNLayer*>::iterator iter_layer = nnet.begin(); iter_layer != nnet.end(); iter_layer++)
	{
		if (iter_layer == nnet.begin())
			(*iter_layer)->setConnection(NULL, *(iter_layer+1));
		else if((*iter_layer) == nnet.back())
			(*iter_layer)->setConnection(*(iter_layer-1), NULL);
		else
			(*iter_layer)->setConnection(*(iter_layer-1), *(iter_layer+1));
	}
}

void ITrain::train()
{
	int T = 50000;

	while(--T)
	{	
		int idx = (nData-1) * rand() / RAND_MAX;

		double *px = data + idx * nDim;
		int *py = label + idx;

		IMatrix *x = new IMatrix(3);
		IMatrix *y = new IMatrix(2);

		double *xn = x->getMatrix();
		double *yn = y->getMatrix();

		*xn = 1;
		for(int k = 0; k < nDim; k++)
			xn[k+1] = px[k];
		*yn = 0;
		*(yn+1) = (double)*py;

		int size = nnet.size();

		nnet[0]->setData(x, NULL);
		nnet[size-1]->setData(NULL, y);
		
		for(int i = 0; i < size; i++)
			nnet[i]->forward();
		
		for(int i = size-1; i >= 0; i--)
		{
			nnet[i]->backpopgation();
			//nnet[i]->gradientChecking();
		}

		for(int i = size-1; i >= 0; i--)
			nnet[i]->updateWeights();				

		delete x;
		delete y;
	}
	test_Ein();
	test_Eout();
}

void ITrain::test_Eout()
{

	int nerror = 0;
	double cost = 0.0;

	double *px = datat;
	int *py = labelt;

	for(int i = 0; i < nDatat; i++)
	{
		printf("\r%d%%", i * 100 / nDatat);
		
		IMatrix *x = new IMatrix(3);
		IMatrix *y = new IMatrix(2);

		double *xn = x->getMatrix();
		double *yn = y->getMatrix();

		*xn = 1;
		for(int k = 0; k < nDimt; k++)
			xn[k+1] = px[k];
		
		*yn = 0;
		*(yn+1) = (double)*py;

		int size = nnet.size();

		nnet[0]->setData(x, NULL);
		nnet[size-1]->setData(NULL, y);

		for(int i = 0; i < size; i++)
			nnet[i]->forward();
		
		int output_dim = nnet[size-1]->getOutputDim();
		double *output_mat = nnet[size-1]->getOutputs()->getMatrix();	
		
		int pre = output_mat[1] > 0 ? 1 : -1;

		if(pre != *py)
			nerror++;
		
		double J = *py - output_mat[1];
		J *= J;

		cost += J; 

		px += nDimt;
		py++;

		delete x;
		delete y;
	}

	printf("\rEout::Error Rate = %d/%d, %0.6f, Cost = %f\n", nerror, nDatat, (float)nerror / nDatat, cost);
}

void ITrain::test_Ein()
{

	int nerror = 0;
	double cost = 0.0;

	double *px = data;
	int *py = label;

	for(int i = 0; i < nData; i++)
	{
		printf("\r%d%%", i * 100 / nData);

		IMatrix *x = new IMatrix(3);
		IMatrix *y = new IMatrix(2);

		double *xn = x->getMatrix();
		double *yn = y->getMatrix();

		*xn = 1;
		for(int k = 0; k < nDim; k++)
			xn[k+1] = px[k];

		*yn = 0;
		*(yn+1) = (double)*py;

		int size = nnet.size();

		nnet[0]->setData(x, NULL);
		nnet[size-1]->setData(NULL, y);

		for(int i = 0; i < size; i++)
			nnet[i]->forward();

		int output_dim = nnet[size-1]->getOutputDim();
		double *output_mat = nnet[size-1]->getOutputs()->getMatrix();	

		int pre = output_mat[1] > 0 ? 1 : -1;

		if(pre != *py)
			nerror++;

		double J = *py - output_mat[1];
		J *= J;

		cost += J; 

		px += nDim;
		py++;

		delete x;
		delete y;
	}

	printf("\rEin::Error Rate = %d/%d, %0.6f, Cost = %f\n", nerror, nData, (float)nerror/nData, cost);
}