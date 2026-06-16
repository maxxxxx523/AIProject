/**
 * Prim 算法 — 最小生成树 (MST)
 *
 * 思想：从一个顶点开始，每次选择连接"当前已选集合"与"未选集合"的最小权重边，
 *       将对应顶点加入集合，直到所有顶点都加入。
 *
 * 时间复杂度：O((V+E) log V)  使用优先队列优化
 */

#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

typedef pair<int, int> pii;   // {权重, 顶点}

int prim(int n, vector<vector<pii>>& graph) {
    vector<bool> inMST(n, false);    // 是否已在生成树中
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    pq.push({0, 0});  // 从顶点 0 开始，权重 0
    int totalWeight = 0;
    int edgesUsed = 0;

    while (!pq.empty() && edgesUsed < n) {
        auto [w, u] = pq.top();
        pq.pop();

        if (inMST[u]) continue;  // 已经在 MST 中，跳过

        inMST[u] = true;
        totalWeight += w;
        edgesUsed++;

        // 松弛 u 的所有邻居
        for (auto& [v, weight] : graph[u]) {
            if (!inMST[v]) {
                pq.push({weight, v});
            }
        }
    }
    return totalWeight;
}

// ---- 测试 ----
int main() {
    int n = 5;   // 顶点数
    vector<vector<pii>> graph(n);

    // 无向图，每条边在两个方向上都加
    auto addEdge = [&](int u, int v, int w) {
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
    };
    addEdge(0, 1, 2);
    addEdge(0, 3, 6);
    addEdge(1, 2, 3);
    addEdge(1, 3, 8);
    addEdge(1, 4, 5);
    addEdge(2, 4, 7);
    addEdge(3, 4, 9);

    int mstWeight = prim(n, graph);
    cout << "MST 总权重: " << mstWeight << endl;  // 期望: 16
    return 0;
}
