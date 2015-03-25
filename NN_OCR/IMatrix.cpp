#include "IMatrix.h"

IMatrix::IMatrix(int r, int c)
{
	this->rows = r;
	this->cols = c;
	mat = new double[r*c];
	memset(mat, 0, sizeof(double)*r*c);
}

IMatrix::IMatrix(int r)
{
	this->rows = r;
	this->cols = 1;
	mat = new double[r];
	memset(mat, 0, sizeof(double)*r);
}

IMatrix::~IMatrix()
{
	if(mat != NULL) {delete[] mat; mat = NULL;}
}

