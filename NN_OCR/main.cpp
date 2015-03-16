#include "NNet.h"
#include <stdio.h>
void main()
{
	NNet m;
	m.ImportTrainingData("data/train.dat");
	m.ImportTestingData("data/test.dat");
	float sum = 0.0f;
	for(int i  = 0; i < 500; i++)
	{
		sum += m.train();
		printf("round %d Eout %f\n", i+1, sum / (i+1));
	}	
	printf("Eout %f\n", sum/500);
}