#include<bits/stdc++.h>
using namespace std;

int show()
{
	int n;
	cin>>n;
	vector<int>vec(n);
	for(int i=0;i<n;i++)
	{
		cin>>vec[i];
	}
	int target;
	cin>>target;
	int left=0,right=vec.size()-1;
	
	while(left<=right)
	{
		int mid=left+(right-left)/2;
		if(vec[mid]==target)
		{
			return mid;
		}
		else if(target<vec[mid])
		{
			right=mid-1;
		}
		else
		{
			left=mid+1;
		}	
	}
	return -1;
}
int main()
{
	cout<<show()<<endl;
	return 0;
	
}
