#include <bits/stdc++.h>
using namespace std;

int main()
{
    int n;
    cin >> n;
    vector<int> vec(n);
    vector<vector<int>> subs;
    for (int i = 0; i < n;++i)
    {
        cin >> vec[i];
    }
    for (int mark = 0; mark < (1 << n);++mark)
    {
        vector<int> sub;
        for (int i = 0; i < n;++i)
        {
            if(mark&(1<<i))
                sub.push_back(vec[i]);
        }
        subs.push_back(sub);
    }
        return 0;
}