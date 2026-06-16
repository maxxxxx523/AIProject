#include <bits/stdc++.h>
using namespace std;

int Partition(vector<int>& arr, int l, int r)
{
    // 随机选择 pivot，避免最坏情况 O(n²)
    int k = l + rand() % (r - l + 1);
    swap(arr[k], arr[r]);
    int pivot = arr[r];
    int i = l - 1;  // i 指向小于 pivot 的区域的最后一个元素
    
    for (int j = l; j < r; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[r]);
    return i + 1;
}

void quicksort(vector<int>& arr, int l, int r)
{
    if (l < r)
    {
        int i = Partition(arr, l, r);
        quicksort(arr, l, i - 1);
        quicksort(arr, i + 1, r);
    }
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));
    vector<int> vec = {8, 10, 54, 16, 14, 49, 26};
    quicksort(vec, 0, vec.size() - 1);
    for (auto it : vec)
        cout << it << " ";
    cout << endl;
    return 0;
}