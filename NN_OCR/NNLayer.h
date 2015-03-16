#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct IMatrix 
{
public:
	IMatrix(int r, int c)
	{
		this->r = r;
		this->c = c;
		m = new float[r*c];
	}

	IMatrix(int r)
	{
		this->r = r;
		this->c = 1;
		m = new float[r];
	}

	~IMatrix()
	{
		if(m != NULL) {delete[] m; m = NULL;}
	}

	int getR(){return r;}
	int getC(){return c;}
	float *getMatrix(){return m;}

private:
	int r;
	int	c;
	float *m;
};

// A layer includes weights and neurons
// this class only stores the related parameters
// calculations are left to others
class NNLayer
{
public:
	NNLayer(int input_dim, int output_dim);
	~NNLayer(void);
	
	int getLayerDim(){return this->layer_dim;}
	IMatrix* getTheta(){return theta;}
	IMatrix* getS(){return s;}
	IMatrix* getX(){return x;}
	IMatrix* getWeights(){return w;}

private:
	// prev_dim, curr_dim does not count the bias as one dim
	int layer_dim;
	IMatrix *theta;
	IMatrix *s;
	IMatrix *x;
	IMatrix *w;
};
