#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
using namespace std;

fstream fp;//声明一个文件流对象
ifstream infile;
int main()
{
	fp.open("C:\\Users\\cbc\\Desktop\\1.txt", ios::out);
	fp << "helloworld" << endl;
	cout << "OK" << endl;
	fp.close();
	infile.open("C:\\Users\\cbc\\Desktop\\1.txt", ios::in);
	string str;
	infile >> str;
	cout << "infile:" << str << endl;
	return 0;
}