#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    static bool cmp(const vector<int> &v1, const vector<int> &v2)
    {
        if (v1[0] != v2[0])
            return v1[0] < v2[0];
        else
            return v1[1] < v2[1];
    }
    vector<vector<int>> merge(vector<vector<int>> &intervals)
    {
        vector<vector<int>> ans;
        sort(intervals.begin(), intervals.end(), cmp);
        int n = intervals.size();
        int start = intervals[0][0], end = intervals[0][1];
        for (int i = 1; i < n; i++)
        {
            if (end >= intervals[i][0])
            {
                if (end < intervals[i][1])
                    end = intervals[i][1];
            }
            else
            {
                ans.push_back({start, end});
                start = intervals[i][0];
                end = intervals[i][1];
            }
        }
        ans.push_back({start, end});
        return ans;
    }
};