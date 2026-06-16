#include <iostream>
using namespace std;
int main ()
{
	int fib_1=1;
	int fib_2=1;
	int n;
	cin>>n;
	for(int i = 3; i<= n;i++)
	{
		int temp=fib_1+fib_2;
		fib_1=fib_2;//倒数第一个数变成了倒数第二个数 
		fib_2=temp;//倒数第一个数赋值为前面两个数数的和 
	}
	cout<<"第n个斐波那契数为"<<fib_2<<endl;
	
	
	
	return 0;
}
