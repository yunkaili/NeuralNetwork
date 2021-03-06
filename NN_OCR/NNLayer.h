#ifndef _NNLAYER_HEAD_H
#define _NNLAYER_HEAD_H

#include "IMatrix.h"
#include <stdlib.h>
#include <string>

enum TRANSFORMATION
{
	SIMGMOID,
	TANH,
	RELU	// scores should be the output of the layer
};

enum COSTFUNCTION
{
	NORM1,
	NORM2,
	LOGLIKEHOOD,
	SOFTMAX,
	LOGISTIC_REGRESSION
};



// A layer includes weights and neurons
class NNLayer
{
public:
	NNLayer(int inputs_dim, int outputs_dim, double initW, double lr, std::string layer_name);
	~NNLayer(void);
	
	/************************************************************************/
	/*                        Config Functions                              */
	/************************************************************************/

	   void setConnection(NNLayer *prev, NNLayer *next);
	   void setData(IMatrix *data, IMatrix *lbl);
 
		int getInputDim() {return inputs_dim;}
		int getOutputDim(){return outputs_dim;}
	NNLayer *getPrevLayer(){return prev_layer;}
	NNLayer *getNextLayer(){return next_layer;}
	IMatrix *getData()	  {return data;}
	IMatrix *getLabel()	  {return lbl;}
	IMatrix *getInputs()  {return inputs;}
	IMatrix *getOutputs() {return outputs;}
	IMatrix *getGrads()	  {return grads;} 
	IMatrix *getScores()  {return scores;}
	IMatrix *getWeights() {return weights;}

	/************************************************************************/
	/*                       Training Functions                             */
	/************************************************************************/

	void feedforward();
	void backpopgation();
	void updateWeights();
	void gradientChecking();

private:

	/************************************************************************/
	/*                           PARAMETERS                                 */
	/************************************************************************/
	std::string Layer_Name;
	// training option
	int	transf;	// Transformation
	int costf;	// Cost Function

	bool gradient_check_enable;
	bool cuda_boost;

	// both inputs and outputs include the bias
	int inputs_dim;
	int outputs_dim;
	double lr;
	
	// Regularization
	double lambda;
	
	IMatrix *inputs; // DO NOT DELETE. because it points to the previous outputs.
	IMatrix *weights;
	IMatrix *scores;
	IMatrix *outputs;	
	IMatrix *grads;

	NNLayer *prev_layer;
	NNLayer *next_layer;
	
	IMatrix *data;
	IMatrix *lbl;

	/************************************************************************/
	/*                           FUNCTIONS                                  */
	/************************************************************************/
	 void activation_function();
	//double gradient_activation_function(double scores);
	 double gradient_activation_function(int idx);
	double getCost();
};

#endif