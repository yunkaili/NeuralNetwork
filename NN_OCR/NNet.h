#pragma once

class NNet
{
public:
	NNet(void);
	~NNet(void);
	void ImportTrainingData(char *dataFile);
	void ImportTestingData(char *dataFile);
	double train();
	void forward(double *w1, double *w2, double *xn, int yn, double &pre_lbl);
	double errorSum(double *w1, double *w2);

	void gradientChecking(double *w1, double *w2, double *theta1, double theta2, double *xn, int yn);
	double NNCost(double *w1, double *w2, double *xn, int yn);

	double train2();
	void forward2(double *w1, double *w2, double *w3, double *xn, int yn, double &pre_lbl);
	double errorSum2(double *w1, double *w2, double *w3);
	void gradientChecking2(double *w1, double *w2, double *w3, 
		double *theta1, double *theta2, double theta3, double *xn, int yn);
	double NNCost2(double *w1, double *w2, double *w3, double *xn, int yn);
	
	template<class T>
	void ptrCheck(T *start, T *ptr, T val_limit, int dist_limit)
	{
		if(ptr-start > dist_limit)
			printf("ERROR!\ndistance to start pointer is %d\n", ptr - start);
		
		if(*ptr > 1000 || *ptr < -1000)
			printf("\nERROR!\nptr value cross line\n");
			
	}

	template<class T>
	int sign(T val)
	{
		if(val <= 0) return -1;
		else return 1;
	}

private:
	double *data;
	int *label;
	int nData;
	int nCls;
	int nDim;

	double *datat;
	int *labelt;
	int nDatat;
	int nClst;
	int nDimt;
};
