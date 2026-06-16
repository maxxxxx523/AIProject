#include <bits/stdc++.h>
using namespace std;
#define maxn 1000010
#define ull unsigned long long
#define B 271

ull Power[maxn];
ull Hash[maxn];

void init(int n, string s)
{
    Power[0] = 1;
    Hash[0] = s[0];
    for (int i = 0; i < s.size(); ++i)
    {
        Hash[i] = Hash[i - 1] * B + s[i];
        Power[i] = Power[i - 1] * B;
    }
}
ull get(int l, int r)
{
    if (l == 0)
        return Hash[r];
    return Hash[r] - Hash[l - 1] * Power[r - l + 1];
}