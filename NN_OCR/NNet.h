#pragma once

class NNet
{
public:
	NNet(void);
	~NNet(void);
	void ImportTrainingData(char *dataFile);
	void ImportTestingData(char *dataFile);
	float train();
	void forward(float *w1, float *w2, float *xn, int yn, float &pre_lbl);
	float errorSum(float *w1, float *w2);

	float train2();
	void forward2(float *w1, float *w2, float *w3, float *xn, int yn, float &pre_lbl);
	float errorSum2(float *w1, float *w2, float *w3);

	void gradientChecking();
	float NNCost(float *w1, float *w2, float *w3);
	
	template<class T>
	void ptrCheck(T *start, T *ptr, T val_limit, int dist_limit)
	{
		if(ptr-start > dist_limit)
			printf("ERROR!\ndistance to start pointer is %d\n", ptr - start);
		
		if(*ptr > 1000 || *ptr < -1000)
			printf("\nERROR!\nptr value cross line\n");
		
		
	}

private:
	float *data;
	int *label;
	int nData;
	int nCls;
	int nDim;

	float *datat;
	int *labelt;
	int nDatat;
	int nClst;
	int nDimt;
};
