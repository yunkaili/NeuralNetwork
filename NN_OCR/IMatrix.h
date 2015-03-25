#ifndef _IMATRIX_HEAD_H
#define _IMATRIX_HEAD_H

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

class IMatrix 
{
public:
	IMatrix(int r, int c);
	IMatrix(int r);
	~IMatrix();

	static void initRandom(IMatrix *matrix, double bound)
	{
		//srand(clock());
		int r = matrix->rows;
		int c = matrix->cols;
		double *pmat = matrix->getMatrix();
		int len = r*c;
		for(int idx = 0; idx < len; idx++)
			pmat[idx] = (double)rand() * 2 * bound / RAND_MAX - bound;
	}

	int  getRows(){ return rows;}
	int  getColcs(){ return cols;}
	double *getMatrix(){ return mat;}
	void  setVal(int r, int c, double val){	mat[r*this->cols+c] = val;}
	double	 getVal(int r, int c){ return mat[r*this->cols+c];}

	void clear(){rows = 0;cols = 0;if(mat){delete[]mat; mat = NULL;}}
	void copy(IMatrix *that)
	{
		this->clear();
		this->rows = that->getRows();
		this->cols = that->getColcs();
		int len = this->rows*this->cols;
		this->mat = new double[len];
		memcpy(this->mat, that->getMatrix(), sizeof(double)*len);
	}

private:
	int rows;
	int	cols;
	double *mat;

};

#endif