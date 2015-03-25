#pragma once
#include "NNLayer.h"
#include <stdio.h>
#include <assert.h>

NNLayer::NNLayer(int inputs_dim, int outputs_dim, double initW, double lr, std::string layer_name)
{
	this->inputs_dim = inputs_dim;
	this->outputs_dim = outputs_dim;

	this->weights	= new IMatrix(outputs_dim-1, inputs_dim);
	IMatrix::initRandom(this->weights, initW);
	this->scores	= new IMatrix(1, outputs_dim-1);
	this->outputs	= new IMatrix(1, outputs_dim);
	this->grads		= new IMatrix(1, outputs_dim-1);

	this->lr = lr;

	this->Layer_Name = layer_name;

	data = NULL;
	lbl = NULL;
	prev_layer = NULL;
	next_layer = NULL;
	inputs = NULL;

}

NNLayer::~NNLayer(void)
{
	inputs = NULL;
	outputs = NULL;
	data = NULL;
	lbl = NULL;
	prev_layer = NULL;
	next_layer = NULL;
	if(weights != NULL) {delete weights; weights = NULL;}
	if(scores != NULL) {delete scores; scores = NULL;}
	if(outputs != NULL) {delete outputs; outputs = NULL;}
	if(grads != NULL) {delete grads; grads = NULL;}
}

void NNLayer::forward()
{
	double *p_inputs;
	if (data)
		p_inputs = data->getMatrix();
	else
		p_inputs = inputs->getMatrix();

	double *p_weights = weights->getMatrix();
	double *p_scores = scores->getMatrix();

	for(int i = 0; i < outputs_dim - 1; i++, p_weights += inputs_dim)
	{
		double sum = 0.0;
		for(int j = 0; j < inputs_dim; j++)
			sum += p_weights[j] * p_inputs[j];
		p_scores[i] = sum;
	}
	
	activation_function();
}

void NNLayer::activation_function()
{
	// using hyperbolic tangent
	double *p_scores = scores->getMatrix();
	double *p_outputs = outputs->getMatrix();
	p_outputs[0] = 1;
	for(int i = 1; i < outputs_dim; i++)
		p_outputs[i] = tanh(p_scores[i-1]);
}

void NNLayer::backpopgation()
{
	double *p_grads = grads->getMatrix();
	double *p_scores= scores->getMatrix();

	if(!next_layer)// output layer
	{
		double *p_outputs = outputs->getMatrix();
		double *p_lbls = lbl->getMatrix();
		
		// theta_last = -2 * (yn - x_last) * x_last'
		for(int i = 1; i < outputs_dim; i++)
			p_grads[i-1] = -2 * (p_lbls[i] - p_outputs[i]) / (cosh(p_scores[i-1])*cosh(p_scores[i-1]));
	}
	else
	{
		double *p_grads_next = next_layer->getGrads()->getMatrix();
		int outputs_dim_next= next_layer->getOutputDim();

		// theta = sum(theta_next * w_next * s'_current);
		for(int i = 1; i < outputs_dim; i++)
		{
			double *p_weights_next = next_layer->getWeights()->getMatrix() + i;

			double sum = 0.0;
			for(int k = 0; k < outputs_dim_next-1; k++, p_weights_next += outputs_dim)
				sum += p_grads_next[k] * (*p_weights_next) / (cosh(p_scores[i-1])*cosh(p_scores[i-1]));

			p_grads[i-1] = sum;
		}
	}
}

void NNLayer::updateWeights()
{
	double *p_weights = weights->getMatrix();
	double *p_grads   = grads->getMatrix();
	double *p_inputs;

	if (data)
		p_inputs = data->getMatrix();
	else
		p_inputs  = inputs->getMatrix();

	// wij = wij - lr * theta * x
	for(int i = 0; i < outputs_dim-1; i++, p_weights += inputs_dim)
	{
		double grad = p_grads[i];
		for(int k = 0; k < inputs_dim; k++)
			p_weights[k] -= lr * grad * p_inputs[k];
	}
}

double NNLayer::getCost()
{
	double J = 0.0;
	forward();
	if (next_layer)
	{
		J = next_layer->getCost();	
	}
	else
	{
		double *p_lbl = lbl->getMatrix();
		double *p_outputs = outputs->getMatrix();

		for(int i = 1; i < outputs_dim; i++)
		{
			J = p_lbl[i] - p_outputs[i];
			J *= J;
		}
	}

	return J;
}

void NNLayer::gradientChecking()
{
	double epsilon = 0.0001;

	double *p_weights = weights->getMatrix();
	double *p_grads = grads->getMatrix();
	double *p_inputs;
	if (prev_layer)
		p_inputs = inputs->getMatrix();
	else
		p_inputs = data->getMatrix();

	for(int i = 0; i < outputs_dim-1; i++)
	{
		for(int k = 0; k < inputs_dim; k++)
		{
			double temp = *p_weights;
			*p_weights = temp + epsilon;
			double Jplus = getCost();
			*p_weights = temp - epsilon;
			double Jminus = getCost();
			double grad0 = (Jplus - Jminus) / (2*epsilon);
			double grad1 = p_grads[i]*p_inputs[k];
			double diff = abs(grad0-grad1);
			if(diff > epsilon)
				printf("%s inputs_dim = %d outputs_dim = %d\nnumerical=%0.6f descent=%0.6f diff=%0.6f\n", 
						Layer_Name.c_str(), k, i, grad0, grad1, diff);
			*p_weights = temp;
			p_weights++;
		}
	}
}

void NNLayer::setConnection(NNLayer *prev, NNLayer *next)
{
	if(prev && data)
	{
		printf("NNLayer::setPrevLayer	Input layer does not have previous layer or Hidden layer does not need inputs\n");
		assert(0);
	}

	if(next && lbl)
	{
		printf("NNLayer::setNextLayer Output layer does not have next layer or Hidden layer does not need lbl\n");
		assert(0);
	}
	
	if(prev)
	{
		prev_layer = prev;
		inputs = prev_layer->getOutputs();
	}

	if(next)
	{
		next_layer = next;
	}
}

void NNLayer::setData(IMatrix *data, IMatrix *lbl)
{
	this->data = data;
	this->lbl = lbl;

	if(prev_layer && data)
	{
		printf("NNLayer::setPrevLayer	Input layer does not have previous layer or Hidden layer does not need inputs\n");
		assert(0);
	}

	if(next_layer && lbl)
	{
		printf("NNLayer::setNextLayer Output layer does not have next layer or Hidden layer does not need lbl\n");
		assert(0);
	}

}

void NNLayer::setTransf(int idx)
{
	switch (idx)
	{
		case 0:
			transf = "sigmoid";
			break;
		case 1:
		default:
			transf = "tanh"
			break;
	}
}