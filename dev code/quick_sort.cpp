/**
 * 快速排序 (Quick Sort)
 * 时间复杂度: 平均 O(n log n) / 最坏 O(n²)  空间复杂度: O(log n) 栈深度  不稳定排序
 *
 * 思想：每次选一个 pivot（基准），将数组划分为 小于pivot | pivot | 大于pivot，
 *       再递归排序左右两部分。
 */

#include <iostream>
#include <vector>
using namespace std;

// 分区函数：返回 pivot 最终位置
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];     // 选最右元素为基准
    int i = low;               // i 指向"小于区"的末尾

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            swap(arr[i], arr[j]);
            i++;
        }
    }
    swap(arr[i], arr[high]);    // pivot 归位
    return i;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low >= high) return;

    int pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
}

// ---- 测试 ----
int main() {
    vector<int> arr = {38, 27, 43, 3, 9, 82, 10};
    quickSort(arr, 0, arr.size() - 1);

    for (int x : arr) cout << x << " ";
    cout << endl;
    return 0;
}
