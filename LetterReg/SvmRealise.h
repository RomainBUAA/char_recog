#ifndef CLASS_SVM_REALISE
#define CLASS_SVM_REALISE

#include <string>
#include <vector>
#include "svm.h"
#include <iostream>
using namespace std;
class SvmData;
class SvmRealise
{
private:
	svm_problem svm_prob;
	svm_parameter param;
	vector<SvmData> svm_data;
	svm_node* test_data;
	double *test_label;
	svm_model *model;
	void get_parse_data(const vector<SvmData>::iterator iter, svm_node* data, int i, int feature_num);

public:
	
	void set_svm_data(vector<SvmData> & _svm_data)
	{
		svm_data = _svm_data;
	}
	void parse_class_data();
	void set_svm_prob();
	void set_svm_param();
	void train_model();
	void predict();
};
class SvmData
{
public:
	char label;
	vector<double> value_vec;
};

#endif // !CLASS_SVM_REALISE