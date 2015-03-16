#include "NNLayer.h"

NNLayer::NNLayer(int input_dim, int output_dim)
{
	this->layer_dim = output_dim;

//	theta = new float[layer_dim-1];
//	s = new float[output_dim];
//	x = new float[input_dim+1];
//	w = new float[(input_dim+1) * output_dim];

	theta = new IMatrix(layer_dim-1, 1);
	s = new IMatrix(output_dim);
	x = new IMatrix(input_dim+1);
	w = new IMatrix(input_dim+1, output_dim);

	int count = (input_dim+1)*output_dim;
	float *pw = w->getMatrix();
	for(int i = 0; i < count; i++)
		pw[i] = (float)rand() / RAND_MAX;

}

NNLayer::~NNLayer(void)
{
	if(theta != NULL){delete[] theta; theta = NULL;}
	if(s != NULL){delete[] s; s = NULL;}
	if(x != NULL){delete[] x; x = NULL;}
	if(w != NULL){delete[] w; w = NULL;}
}
