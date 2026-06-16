# include "yang.h"
#include <string>
using namespace std;

void PrintBlank(int k)
{
    k -= 1;
    for (int i = 0; i < k; i++)
    {
        cout << " ";
    }
}

int Combineni(int n, int i)
{
    if (i == 1)
    {
        return n;
    }
    else if (i == 0)
    {
        return 1;
    }
    else if (i == 0 && n == 0)
    {
        return 1;
    }
    else return (Combineni(n, i - 1) * (n - i + 1) / i);
}
void PrintCombine(int i)
{
    i -= 1;
    for (int j = 0; j <= i; j++)
    {
        cout << Combineni(i, j) << " ";
    }
}

void printf(char a)
{
    cout << a;
}
int main()
{
    int n;
    cin >> n;
    for (int i = 0; i < n; i++)
    {
        PrintBlank(n - i);      // 打印前导空格
        PrintCombine(i);      // 打印第i行的系数
        printf("\n");         // 换行
    }
    return 0;
}
