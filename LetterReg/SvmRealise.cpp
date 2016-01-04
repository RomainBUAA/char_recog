#include "SvmRealise.h"
#include "svm.h"

void SvmRealise::parse_class_data()
{
	vector<SvmData>::iterator iter;
	int i = 0;
	int feature_num = svm_data[0].value_vec.size();
	cout << "feature_num is" << feature_num;
	//the number of feature
	int len_test_data = svm_data.size() - svm_prob.l;
	test_label = new double[len_test_data];

	svm_node* feature_space = new svm_node[(feature_num + 1)*svm_prob.l];//
	test_data = new svm_node[(feature_num + 1)];

	for (iter = svm_data.begin(); iter!=svm_data.begin() + svm_prob.l; iter++)
	{

			get_parse_data(iter, feature_space, i, feature_num);
			//cout << (*iter).label;
			//cout << (*iter).label;
			//cout << svm_prob.x[i]<<endl;
			svm_prob.y[i] = (*iter).label;
			svm_prob.x[i] = &feature_space[(feature_num + 1)*i];
			
			i++;


	}



}

void SvmRealise::get_parse_data(const vector<SvmData>::iterator _iter, svm_node* data,int i , int feature_num)
{
	int j = 0;
	for (vector<double>::iterator value_iter = (*_iter).value_vec.begin(); value_iter != (*_iter).value_vec.end(); value_iter++)
	{
		
			data[i*(feature_num + 1) + j].value = *value_iter;
			data[i*(feature_num + 1) + j].index = j + 1;
			//cout << *value_iter<<" ";
			j++;
	}
	//cout << endl;
	data[i*(feature_num + 1) + j].index = -1;
	//cout << "data is " << data<<endl;

}
void SvmRealise::set_svm_prob()

{
	svm_prob.l = 4 / 5.0*svm_data.size();
	svm_prob.x = new svm_node *[svm_prob.l];
	svm_prob.y = new double[svm_prob.l];


}
void SvmRealise::predict()
{
	vector<SvmData>::iterator iter;
	int i = 0;
	int feature_num = svm_data[0].value_vec.size();//the number of feature
	int len_test_data = svm_data.size() - svm_prob.l;
	char *output_label = new char[len_test_data];
	int sum = 0;
	for (iter = svm_data.begin() + svm_prob.l;iter!=svm_data.end(); iter++)
	{

		get_parse_data(iter, test_data, 0, feature_num);
		test_label[i] = (*iter).label;
		//for (vector<int>::iterator int_iter = (*iter).value_vec.begin(); int_iter != (*iter).value_vec.end(); int_iter++)
		//	cout<<(*int_iter)<<" ";
		//cout << endl;
		output_label[i] = svm_predict(model, test_data);
		if ((char)test_label[i] == output_label[i])
			sum++;
		cout << "test label is " << (char)test_label[i] << " ";
		cout << "output label is " << output_label[i] << endl;

		i++;

	}
	cout << "accurary is " << sum*1.0 * 100 / len_test_data << "%" << endl;

}
void SvmRealise::train_model()
{
	model = svm_train(&svm_prob, &param);

}
void SvmRealise::set_svm_param()
{
	param.svm_type= C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.03846;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 10;
	param.eps = 1e-5;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	
}



