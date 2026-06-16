#include <bits/stdc++.h>
using namespace std;
/*123
   12
----------
   */
int main()
{
	string num1, num2;
	cin >> num1 >> num2;
	vector<int> n1, n2;
	for (int i = num1.size() - 1; i >= 0; i--)
	{
		n1.push_back(num1[i] - '0');
	}
	for (int i = num2.size() - 1; i >= 0; i--)
	{
		n2.push_back(num2[i] - '0');
	}
	vector<int> num(n1.size() + n2.size());
	for (int i = 0; i < n1.size(); i++)
	{
		for (int j = 0; j < n2.size(); j++)
		{
			num[i + j] += n1[i] * n2[j];
		}
	}
	for (int i = 0; i < num.size() - 1; i++)
	{
		num[i + 1] += num[i] / 10;
		num[i] = num[i] % 10;
	}
	reverse(num.begin(), num.end());
	int i = 0;
	for (i = 0; i < num.size(); i++)
	{
		if (num[i] != 0)
			break;
	}
	for (int j = i; j < num.size(); j++)
	{
		cout << num[j];
	}
	return 0;
}