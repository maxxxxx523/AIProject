/**
 * 迪杰斯特拉算法 (Dijkstra)
 *
 * 用途：求单源最短路径 — 从起点 src 到图中所有其他顶点的最短距离。
 * 限制：边权必须非负。
 *
 * 时间复杂度：O((V+E) log V)  使用优先队列优化
 */

#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

typedef pair<int, int> pii;   // {距离, 顶点}

// 邻接表存图: graph[u] = { {v, w}, ... }
vector<int> dijkstra(int n, vector<vector<pii>>& graph, int src) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;

    priority_queue<pii, vector<pii>, greater<pii>> pq;  // 小顶堆
    pq.push({0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d != dist[u]) continue;  // 过时的条目，跳过

        for (auto& [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

// ---- 测试 ----
int main() {
    int n = 5;   // 顶点数
    vector<vector<pii>> graph(n);

    // 加边: u -> v，权重 w
    graph[0].push_back({1, 10});
    graph[0].push_back({3, 5});
    graph[1].push_back({2, 1});
    graph[1].push_back({3, 2});
    graph[2].push_back({4, 4});
    graph[3].push_back({1, 3});
    graph[3].push_back({2, 9});
    graph[3].push_back({4, 2});
    graph[4].push_back({0, 7});
    graph[4].push_back({2, 6});

    int src = 0;
    vector<int> dist = dijkstra(n, graph, src);

    for (int i = 0; i < n; i++)
        cout << src << " -> " << i << " : " << dist[i] << endl;
    return 0;
}
