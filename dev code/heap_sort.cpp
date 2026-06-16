/**
 * 堆排序 (Heap Sort)
 * 时间复杂度: O(n log n)  空间复杂度: O(1)  不稳定排序
 *
 * 思想：利用大顶堆（父节点 ≥ 子节点）进行选择排序。
 *       建堆 → 反复将堆顶（最大值）与末尾交换，再调整堆。
 */

#include <iostream>
#include <vector>
using namespace std;

// 从节点 i 开始向下调整大顶堆，范围 [0, n-1]
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;          // 当前最大节点
    int left   = 2 * i + 1;   // 左孩子
    int right  = 2 * i + 2;   // 右孩子

    if (left  < n && arr[left]  > arr[largest]) largest = left;
    if (right < n && arr[right] > arr[largest]) largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);  // 递归调整被交换的子树
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();

    // 1. 建堆：从最后一个非叶节点开始向上调整
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // 2. 排序：依次将堆顶(最大值)与末尾交换，缩小堆再调整
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);      // 最大值放到已排序区末尾
        heapify(arr, i, 0);        // 对剩余元素重新调整
    }
}

// ---- 测试 ----
int main() {
    vector<int> arr = {38, 27, 43, 3, 9, 82, 10};
    heapSort(arr);

    for (int x : arr) cout << x << " ";
    cout << endl;
    return 0;
}
