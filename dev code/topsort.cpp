#include <bits/stdc++.h>
using namespace std;

class TopSort
{
public:
    int n;
    vector<pair<string, string>> verts;
    vector<vector<int>> graph;
    unordered_map<int, string> names;
    unordered_map<string, int> nums;
    vector<string> ans;
    
    vector<vector<int>> getgraph()
    {
        vector<vector<int>> g(n, vector<int>(n, 0));
        for (auto it : verts)
        {
            g[nums[it.first]][nums[it.second]]++;
        }
        return g;
    }

    void initial(int cn)
    {
        n = cn;
        string n1, n2;
        int cnt = 0;
        for (int i = 0; i < n - 1; ++i)
        {
            cin >> n1 >> n2;
            verts.push_back({n1, n2});

            if (nums.find(n1) == nums.end())
            {
                nums[n1] = cnt;
                names[cnt] = n1;
                cnt++;
            }
            if (nums.find(n2) == nums.end())
            {
                nums[n2] = cnt;
                names[cnt] = n2;
                cnt++;
            }
        }
        graph = getgraph();
    }

    int getin(int k)
    {
        int sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += graph[i][k];
        }
        return sum;
    }

    void topsort()
    {
        queue<int> q;
        vector<int> in_degree;
        for (int i = 0; i < n; ++i)
        {
            in_degree.push_back(getin(i));
        }
        for (int i = 0; i < n ; ++i)
        {
            if (in_degree[i] == 0)
                q.push(i);
        }
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            ans.push_back(names[u]);

            // 将所有邻接点的入度减1
            for (int v = 0; v < n; ++v)
            {
                if (graph[u][v] > 0)
                { // 有边 u->v
                    in_degree[v]--;
                    if (in_degree[v] == 0)
                    {
                        q.push(v);
                    }
                }
            }
        }
    }
    void print()
    {
        for (auto i : ans)
        {
            cout << i << endl;
        }
        cout << endl;
    }
};

int main()
{
    int cn;
    vector<TopSort> ans;
    while (cin >> cn)
    {
        if (cn == 0)
            break;
        TopSort t;
        t.initial(cn);
        t.topsort();
        ans.push_back(t);
    }
    for(auto it:ans)
        it.print();
    return 0;
}