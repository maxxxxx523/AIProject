#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    int maxSubArray(vector<int> &nums)
    {
        int max_sum = nums[0], current_sum = 0;
        for (int i : nums)
        {
            current_sum = max(current_sum + i, i);
            max_sum = max(current_sum, max_sum);
        }
        return max_sum;
    }
};