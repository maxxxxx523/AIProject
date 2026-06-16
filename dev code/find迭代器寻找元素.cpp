#include <iostream>
#include <vector>
using namespace std;

int main()
{
	vector <int>arr;
	int n;
	cin >> n;
	for (int i = 0; i < n; i++)
	{
		int x;
		cin >> x;
		arr.push_back(x);
	}
	auto it = find(arr.begin(), arr.end(), 1);
	if (it != arr.end())
		cout << "ур╣╫1ак" << endl;
	return 0;
}
