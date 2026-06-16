/**
 * 二路归并排序 (Two-way Merge Sort)
 * 时间复杂度: O(n log n)  空间复杂度: O(n)  稳定排序
 *
 * 思想：将数组不断二分，递归排序左右两半，再将两个有序子数组合并。
 */

#include <iostream>
#include <vector>
using namespace std;

// 合并两个有序子数组 arr[l..mid] 和 arr[mid+1..r]
void merge(vector<int>& arr, int l, int mid, int r) {
    vector<int> tmp(r - l + 1);
    int i = l;          // 左半指针
    int j = mid + 1;    // 右半指针
    int k = 0;          // 临时数组指针

    while (i <= mid && j <= r) {
        if (arr[i] <= arr[j])
            tmp[k++] = arr[i++];
        else
            tmp[k++] = arr[j++];
    }

    // 把剩余的元素拷贝进去
    while (i <= mid) tmp[k++] = arr[i++];
    while (j <= r)   tmp[k++] = arr[j++];

    // 写回原数组
    for (k = 0; k < (int)tmp.size(); k++)
        arr[l + k] = tmp[k];
}

// 递归归并排序
void mergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int mid = l + (r - l) / 2;
    mergeSort(arr, l, mid);
    mergeSort(arr, mid + 1, r);
    merge(arr, l, mid, r);
}

// ---- 测试 ----
int main() {
    vector<int> arr = {38, 27, 43, 3, 9, 82, 10};
    mergeSort(arr, 0, arr.size() - 1);

    for (int x : arr) cout << x << " ";
    cout << endl;
    return 0;
}
