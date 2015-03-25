#pragma once
#include "ITrain.h"
#include <time.h>

void main()
{
	srand(clock());
	ITrain m_train;
	
	m_train.ImportTrainingData("data/train.dat");
	m_train.ImportTestingData("data/test.dat");
	for(int i = 0; i < 500; i++)
	{
		m_train.clear_config();
		m_train.config();
		m_train.train();
	}

}