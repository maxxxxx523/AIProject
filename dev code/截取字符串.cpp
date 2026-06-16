#include <iostream>
#include <string>
using namespace std;

int main()
{
	int num;
	cin >> num;
	string num1 = to_string(num);
	string num11 = num1.substr(0, 2);
	cout << num1 << endl;
	cout << num11 << endl;
	return 0; 

}
