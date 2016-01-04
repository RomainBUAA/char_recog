#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>

#include <iostream>
#include <fstream>
#include <string>

#include "SvmRealise.h"
#include <vector>
#include <Windows.h>
#include "svm.h"
#define MAX_CHAR_LEN 256
#define MAX_SINGLE_LEN 100
using namespace std;
using namespace cv;
void read_fail_exit()
{
	cout << "cant open file" << endl;
	exit(1);//with error
}


void store_data_fuc(const string &_path, vector<SvmData> & _svm_data_vec)
{
	
	cout << "start read data" << endl;
	cout << "path is " << _path << endl;
	//stringstream buffer;

	//char *buffer = new char[MAX_CHAR_LEN];
	SvmData *svm_data;
	string buffer;
	double value;
	ifstream file(_path);
	if (!file)
	{
		read_fail_exit();

	}
	int i = 0;
	while (getline(file, buffer))
	{
		i++;
		svm_data = new SvmData;
		//cout << buffer;
		istringstream str_stream(buffer);
		str_stream >> svm_data->label;
		str_stream.ignore(buffer.size(), ',');
		//cout << svm_data.label;
		while (str_stream >> value)
		{

			svm_data->value_vec.push_back(value);
			//cout << value << ' ';
			str_stream.ignore(buffer.size(), ','); 
			
			if (i % 1000 == 0)
			{
	
				cout <<"finish  "<< (i*1.0 /20000*100)<< '%'<<endl;
				cout << svm_data->label;
				cout << value;
			}
		}
		
		//cout << endl;
		_svm_data_vec.push_back(*svm_data);
	}

	file.close();
}


void main()
{

	vector<SvmData> svm_data_vec;
	string path="D:\\project\\LetterReg\\a.data";
	store_data_fuc(path, svm_data_vec);
	SvmRealise svm_realise;
	svm_realise.set_svm_data(svm_data_vec);
	svm_realise.set_svm_prob();
	svm_realise.set_svm_param();
	svm_realise.parse_class_data();
	svm_realise.train_model();
	svm_realise.predict();
	system("pause");

}