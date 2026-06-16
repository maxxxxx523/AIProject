/**
 * Kruskal 算法 — 最小生成树 (MST)
 *
 * 思想：将所有边按权重排序，从小到大依次检查每条边：
 *       如果边的两端不在同一集合中，则选中该边并合并集合。
 *       使用并查集 (Union-Find) 来判断连通性。
 *
 * 时间复杂度：O(E log E)  主要耗时在边的排序
 */

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// ---- 并查集 (Union-Find / Disjoint Set Union) ----
struct DSU {
    vector<int> parent, rank;

    DSU(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);  // 路径压缩
        return parent[x];
    }

    void unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx == ry) return;
        // 按秩合并
        if (rank[rx] < rank[ry])      parent[rx] = ry;
        else if (rank[rx] > rank[ry]) parent[ry] = rx;
        else                          { parent[ry] = rx; rank[rx]++; }
    }
};

// 边的结构体
struct Edge {
    int u, v, w;
    // 用于排序
    bool operator<(const Edge& other) const {
        return w < other.w;
    }
};

pair<int, vector<Edge>> kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());  // 按边权升序排序

    DSU dsu(n);
    vector<Edge> mstEdges;
    int totalWeight = 0;

    for (auto& e : edges) {
        if (dsu.find(e.u) != dsu.find(e.v)) {
            dsu.unite(e.u, e.v);
            mstEdges.push_back(e);
            totalWeight += e.w;
            if ((int)mstEdges.size() == n - 1) break;  // MST 已包含 n-1 条边
        }
    }
    return {totalWeight, mstEdges};
}

// ---- 测试 ----
int main() {
    int n = 5;   // 顶点数
    vector<Edge> edges = {
        {0, 1, 2},
        {0, 3, 6},
        {1, 2, 3},
        {1, 3, 8},
        {1, 4, 5},
        {2, 4, 7},
        {3, 4, 9},
    };

    auto [totalWeight, mstEdges] = kruskal(n, edges);

    cout << "MST 总权重: " << totalWeight << endl;  // 期望: 16
    cout << "MST 边集:" << endl;
    for (auto& e : mstEdges)
        cout << e.u << " - " << e.v << " (w=" << e.w << ")" << endl;
    return 0;
}
