#include <bits/stdc++.h>
using namespace std;

// 边结点 — 邻接表中的边
struct EdgeNode
{
    int dest;
    EdgeNode *next;
    EdgeNode(int d, EdgeNode *n = nullptr) : dest(d), next(n) {}
};

// 顶点结点 — 邻接表的表头
struct VerNode
{
    EdgeNode *adj;
    VerNode() : adj(nullptr) {}
};

// 回路链表结点
struct EulerNode
{
    int ver;
    EulerNode *next;
    EulerNode(int v, EulerNode *n = nullptr) : ver(v), next(n) {}
};

// 回路链表首尾指针
struct CircPtrNode
{
    EulerNode *first;
    EulerNode *last;
    CircPtrNode() : first(nullptr), last(nullptr) {}
};

class EulerGraph
{
private:
    int n_verts;
    vector<VerNode> ver_list;

    // 复制一份邻接表副本
    EulerGraph clone() const
    {
        EulerGraph g;
        g.n_verts = n_verts;
        g.ver_list.resize(n_verts);
        for (int v = 0; v < n_verts; v++)
        {
            EdgeNode *p = ver_list[v].adj;
            EdgeNode **tail = &g.ver_list[v].adj;
            while (p)
            {
                *tail = new EdgeNode(p->dest);
                tail = &((*tail)->next);
                p = p->next;
            }
        }
        return g;
    }

    // 删除无向图中的一条边 (u, v)（同时删除两个方向）
    void removeEdge(int u, int v)
    {
        EdgeNode *p = ver_list[u].adj, *prev = nullptr;
        while (p && p->dest != v)
        {
            prev = p;
            p = p->next;
        }
        if (p)
        {
            if (prev)
                prev->next = p->next;
            else
                ver_list[u].adj = p->next;
            delete p;
        }
        p = ver_list[v].adj;
        prev = nullptr;
        while (p && p->dest != u)
        {
            prev = p;
            p = p->next;
        }
        if (p)
        {
            if (prev)
                prev->next = p->next;
            else
                ver_list[v].adj = p->next;
            delete p;
        }
    }

    // 算法7-20：从给定顶点出发获得一条回路
    CircPtrNode *getCircuit(int start)
    {
        EulerNode *new_node = new EulerNode(start, nullptr);
        CircPtrNode *circuit = new CircPtrNode();
        circuit->first = new_node;
        circuit->last = new_node;
        int head = start;
        EdgeNode *p = ver_list[head].adj;
        while (p)
        {
            int tail = p->dest;
            removeEdge(head, tail);
            new_node = new EulerNode(tail, nullptr);
            circuit->last->next = new_node;
            circuit->last = circuit->last->next;
            if (tail == start)
                break;
            head = tail;
            p = ver_list[head].adj;
        }
        return circuit;
    }

public:
    EulerGraph() : n_verts(0) {}

    void init(int n)
    {
        n_verts = n;
        ver_list.assign(n, VerNode());
    }

    // 添加无向边
    void addEdge(int u, int v)
    {
        ver_list[u].adj = new EdgeNode(v, ver_list[u].adj);
        ver_list[v].adj = new EdgeNode(u, ver_list[v].adj);
    }

    // 算法7-21：求欧拉回路
    CircPtrNode *eulerCircle()
    {
        // 计算每个顶点的度，判断是否存在欧拉回路
        for (int v = 0; v < n_verts; v++)
        {
            EdgeNode *p = ver_list[v].adj;
            int degree = 0;
            while (p)
            {
                degree++;
                p = p->next;
            }
            if (degree % 2 == 1)
                return nullptr;
        }

        EulerGraph tmp_graph = clone();
        CircPtrNode *circuit = tmp_graph.getCircuit(0);
        EulerNode *cp = circuit->first->next;
        while (cp)
        {
            if (tmp_graph.ver_list[cp->ver].adj)
            {
                CircPtrNode *next_circuit = tmp_graph.getCircuit(cp->ver);
                next_circuit->last->next = cp->next;
                cp->next = next_circuit->first->next;
                delete next_circuit->first;
                delete next_circuit;
            }
            cp = cp->next;
        }
        return circuit;
    }

    // 输出回路
    void printCircuit(CircPtrNode *circuit) const
    {
        if (!circuit)
        {
            cout << "NIL" << endl;
            return;
        }
        for (EulerNode *p = circuit->first; p; p = p->next)
        {
            if (p != circuit->first)
                cout << " -> ";
            cout << p->ver;
        }
        cout << endl;
    }

    // 释放回路链表内存
    static void freeCircuit(CircPtrNode *circuit)
    {
        if (!circuit)
            return;
        EulerNode *p = circuit->first;
        while (p)
        {
            EulerNode *next = p->next;
            delete p;
            p = next;
        }
        delete circuit;
    }
};

int main()
{
    EulerGraph g;
    // 示例：3个顶点构成三角形 (0-1-2-0)，存在欧拉回路
    g.init(7);
    g.addEdge(0, 1);
    g.addEdge(0, 6);
    g.addEdge(1, 2);
    g.addEdge(2, 6);
    g.addEdge(2, 3);
    g.addEdge(2, 5);
    g.addEdge(3, 4);
    g.addEdge(4, 5);

    CircPtrNode *circuit = g.eulerCircle();
    g.printCircuit(circuit);
    EulerGraph::freeCircuit(circuit);
    return 0;
}
