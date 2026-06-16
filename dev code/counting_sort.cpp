/**
 * 计数排序 (Counting Sort)
 * 时间复杂度: O(n + k)  空间复杂度: O(k)  稳定排序
 *
 * 前提：待排序元素必须是整数，且范围已知/可控。
 * 思想：统计每个值出现次数 → 计算前缀和确定位置 → 从后往前放入输出数组。
 * 注意：下面实现的是「稳定版」计数排序。
 */

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void countingSort(vector<int>& arr) {
    if (arr.empty()) return;

    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    int range = maxVal - minVal + 1;

    vector<int> count(range, 0);
    vector<int> output(arr.size());

    // 1. 统计频次
    for (int x : arr)
        count[x - minVal]++;

    // 2. 求前缀和 —— count[i] 表示 ≤ (minVal+i) 的元素个数
    for (int i = 1; i < range; i++)
        count[i] += count[i - 1];

    // 3. 从后往前遍历，保证稳定性
    for (int i = (int)arr.size() - 1; i >= 0; i--) {
        int idx = arr[i] - minVal;
        output[count[idx] - 1] = arr[i];
        count[idx]--;
    }

    // 4. 写回原数组
    arr = output;
}

// ---- 测试 ----
int main() {
    vector<int> arr = {4, 2, 2, 8, 3, 3, 1};
    countingSort(arr);

    for (int x : arr) cout << x << " ";
    cout << endl;
    return 0;
}
