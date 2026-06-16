#include <bits/stdc++.h>
using namespace std;

const int MAX_ENTRIES = 4;               // M
const int MIN_ENTRIES = MAX_ENTRIES / 2; // m

struct Rect
{
    int minX, minY, maxX, maxY;

    Rect() : minX(0), minY(0), maxX(0), maxY(0) {}
    Rect(int x1, int y1, int x2, int y2)
        : minX(min(x1, x2)), minY(min(y1, y2)), maxX(max(x1, x2)), maxY(max(y1, y2)) {}

    long long area() const
    {
        return (maxX > minX && maxY > minY)
                   ? (long long)(maxX - minX) * (maxY - minY)
                   : 0;
    }

    // 两个矩形的并集
    Rect unite(const Rect &o) const
    {
        return Rect(min(minX, o.minX), min(minY, o.minY),
                    max(maxX, o.maxX), max(maxY, o.maxY));
    }

    // 两个矩形是否相交
    bool intersects(const Rect &o) const
    {
        return !(minX > o.maxX || maxX < o.minX ||
                 minY > o.maxY || maxY < o.minY);
    }

    // 包含点
    bool containsPoint(int x, int y) const
    {
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }

    // 包含另一个矩形
    bool contains(const Rect &o) const
    {
        return minX <= o.minX && maxX >= o.maxX &&
               minY <= o.minY && maxY >= o.maxY;
    }

    // 扩建到包含另一个矩形的面积增量
    long long enlargement(const Rect &o) const
    {
        Rect u = unite(o);
        return u.area() - area();
    }
};

struct RTreeNode
{
    bool isLeaf;
    vector<Rect> mbr;             // 包围盒数组
    vector<int> data;             // 叶节点数据 (store int)
    vector<RTreeNode *> children; // 内部节点子指针

    RTreeNode(bool leaf) : isLeaf(leaf) {}
};

class RTree
{
private:
    RTreeNode *root;
    int size_; // 数据条目数

    // ---- 辅助: 计算两个矩形之间的距离 (中心距离平方) ----
    static long long centerDistSq(const Rect &a, const Rect &b)
    {
        long long cx1 = (long long)a.minX + a.maxX;
        long long cy1 = (long long)a.minY + a.maxY;
        long long cx2 = (long long)b.minX + b.maxX;
        long long cy2 = (long long)b.minY + b.maxY;
        long long dx = cx1 - cx2;
        long long dy = cy1 - cy2;
        return dx * dx + dy * dy;
    }

    // ---- 线性分裂: 沿最长轴分成两组 ----
    void linearSplit(RTreeNode *node, RTreeNode *&out1, RTreeNode *&out2)
    {
        int n = node->mbr.size();
        // 计算整体包围盒
        Rect bounds = node->mbr[0];
        for (int i = 1; i < n; i++)
            bounds = bounds.unite(node->mbr[i]);

        bool splitByX = (bounds.maxX - bounds.minX) >= (bounds.maxY - bounds.minY);

        // 按坐标排序入口
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        if (splitByX)
        {
            sort(idx.begin(), idx.end(), [&](int a, int b)
                 { return node->mbr[a].minX + node->mbr[a].maxX <
                          node->mbr[b].minX + node->mbr[b].maxX; });
        }
        else
        {
            sort(idx.begin(), idx.end(), [&](int a, int b)
                 { return node->mbr[a].minY + node->mbr[a].maxY <
                          node->mbr[b].minY + node->mbr[b].maxY; });
        }

        out1 = new RTreeNode(node->isLeaf);
        out2 = new RTreeNode(node->isLeaf);
        int half = n / 2;
        for (int i = 0; i < n; i++)
        {
            int j = idx[i];
            if (i < half)
                addEntry(out1, node->mbr[j], node->data[j],
                         node->isLeaf ? nullptr : node->children[j]);
            else
                addEntry(out2, node->mbr[j], node->data[j],
                         node->isLeaf ? nullptr : node->children[j]);
        }
    }

    // ---- 选择一个叶子节点来插入 ----
    RTreeNode *chooseLeaf(const Rect &r)
    {
        RTreeNode *cur = root;
        while (!cur->isLeaf)
        {
            long long bestEnlarge = LLONG_MAX;
            RTreeNode *best = nullptr;
            for (size_t i = 0; i < cur->children.size(); i++)
            {
                long long enl = cur->mbr[i].enlargement(r);
                if (enl < bestEnlarge)
                {
                    bestEnlarge = enl;
                    best = cur->children[i];
                }
            }
            cur = best;
        }
        return cur;
    }

    // ---- 给节点添加条目 ----
    static void addEntry(RTreeNode *node, const Rect &r, int d, RTreeNode *child)
    {
        node->mbr.push_back(r);
        node->data.push_back(d);
        if (!node->isLeaf && child)
            node->children.push_back(child);
    }

    // ---- 重新计算节点的MBR ----
    static Rect computeMBR(RTreeNode *node)
    {
        Rect r = node->mbr[0];
        for (size_t i = 1; i < node->mbr.size(); i++)
            r = r.unite(node->mbr[i]);
        return r;
    }

    // ---- 插入到子树 ----
    RTreeNode *insertRec(RTreeNode *node, const Rect &r, int val)
    {
        if (node->isLeaf)
        {
            addEntry(node, r, val, nullptr);
            if ((int)node->mbr.size() > MAX_ENTRIES)
            {
                RTreeNode *a, *b;
                linearSplit(node, a, b);
                delete node;
                // 创建新的内部节点
                RTreeNode *parent = new RTreeNode(false);
                addEntry(parent, computeMBR(a), 0, a);
                addEntry(parent, computeMBR(b), 0, b);
                return parent;
            }
            return node;
        }

        // 选一个子节点插入
        long long bestEnlarge = LLONG_MAX;
        int bestIdx = 0;
        for (size_t i = 0; i < node->children.size(); i++)
        {
            long long enl = node->mbr[i].enlargement(r);
            if (enl < bestEnlarge)
            {
                bestEnlarge = enl;
                bestIdx = i;
            }
        }

        RTreeNode *newChild = insertRec(node->children[bestIdx], r, val);
        if (newChild != node->children[bestIdx])
        {
            // 子节点分裂了
            node->children[bestIdx] = newChild;
            node->mbr[bestIdx] = computeMBR(newChild);
            // newChild 其实已经是 node->children[bestIdx], 但我们需要判断是否需要进一步分裂
            // 检查 newChild 是否只有一个孩子并且是一个被提升的内部节点
        }
        // 更新该子节点的MBR
        node->mbr[bestIdx] = computeMBR(node->children[bestIdx]);

        if ((int)node->children.size() > MAX_ENTRIES)
        {
            RTreeNode *a, *b;
            linearSplit(node, a, b);
            delete node;
            RTreeNode *parent = new RTreeNode(false);
            addEntry(parent, computeMBR(a), 0, a);
            addEntry(parent, computeMBR(b), 0, b);
            return parent;
        }
        return node;
    }

    // ---- 递归插入（重写，更清晰的版本） ----
    RTreeNode *insertRect(RTreeNode *node, const Rect &r, int val)
    {
        if (node == nullptr)
        {
            node = new RTreeNode(true);
            addEntry(node, r, val, nullptr);
            size_++;
            return node;
        }

        if (node->isLeaf)
        {
            addEntry(node, r, val, nullptr);
            size_++;
            if ((int)node->mbr.size() > MAX_ENTRIES)
            {
                RTreeNode *a, *b;
                linearSplit(node, a, b);
                delete node;
                RTreeNode *parent = new RTreeNode(false);
                addEntry(parent, computeMBR(a), 0, a);
                addEntry(parent, computeMBR(b), 0, b);
                return parent;
            }
            return node;
        }

        // 选最佳子节点
        long long bestEnlarge = LLONG_MAX;
        long long bestArea = LLONG_MAX;
        int bestIdx = 0;
        for (size_t i = 0; i < node->children.size(); i++)
        {
            long long enl = node->mbr[i].enlargement(r);
            long long ar = node->mbr[i].area();
            if (enl < bestEnlarge || (enl == bestEnlarge && ar < bestArea))
            {
                bestEnlarge = enl;
                bestArea = ar;
                bestIdx = i;
            }
        }

        RTreeNode *childResult = insertRect(node->children[bestIdx], r, val);
        node->children[bestIdx] = childResult;
        node->mbr[bestIdx] = computeMBR(childResult);

        if ((int)node->children.size() > MAX_ENTRIES)
        {
            RTreeNode *a, *b;
            linearSplit(node, a, b);
            delete node;
            RTreeNode *parent = new RTreeNode(false);
            addEntry(parent, computeMBR(a), 0, a);
            addEntry(parent, computeMBR(b), 0, b);
            return parent;
        }
        return node;
    }

    // ---- 范围搜索 ----
    void searchRec(RTreeNode *node, const Rect &query, vector<int> &result)
    {
        if (!node)
            return;
        for (size_t i = 0; i < node->mbr.size(); i++)
        {
            if (!node->mbr[i].intersects(query))
                continue;
            if (node->isLeaf)
            {
                result.push_back(node->data[i]);
            }
            else
            {
                searchRec(node->children[i], query, result);
            }
        }
    }

    // ---- 删除辅助: 寻找包含该数据和矩形的叶子 ----
    bool deleteRec(RTreeNode *node, const Rect &r, int val)
    {
        if (!node)
            return false;
        for (size_t i = 0; i < node->mbr.size(); i++)
        {
            if (!node->mbr[i].contains(r))
                continue;
            if (node->isLeaf)
            {
                if (node->mbr[i].minX == r.minX && node->mbr[i].minY == r.minY &&
                    node->mbr[i].maxX == r.maxX && node->mbr[i].maxY == r.maxY &&
                    node->data[i] == val)
                {
                    node->mbr.erase(node->mbr.begin() + i);
                    node->data.erase(node->data.begin() + i);
                    size_--;
                    return true;
                }
            }
            else
            {
                if (deleteRec(node->children[i], r, val))
                {
                    // 更新MBR，若子节点为空则移除
                    if (node->children[i]->mbr.empty())
                    {
                        delete node->children[i];
                        node->children.erase(node->children.begin() + i);
                        node->mbr.erase(node->mbr.begin() + i);
                        node->data.erase(node->data.begin() + i);
                    }
                    else
                    {
                        node->mbr[i] = computeMBR(node->children[i]);
                    }
                    return true;
                }
            }
        }
        return false;
    }

    // ---- 将条目重新插入 (用于处理下溢) ----
    void reinsertEntries(RTreeNode *node)
    {
        // 简化处理: 如果根节点是内部节点且只有一个孩子, 降高度
        (void)node;
    }

    void condenseTree(RTreeNode *node)
    {
        if (node == root)
        {
            if (!node->isLeaf && node->children.size() == 1)
            {
                RTreeNode *newRoot = node->children[0];
                node->children.clear();
                node->mbr.clear();
                node->data.clear();
                delete node;
                root = newRoot;
            }
            return;
        }
    }

    // ---- 打印 ----
    void printRec(RTreeNode *node, int depth)
    {
        if (!node)
            return;
        string indent(depth * 4, ' ');
        for (size_t i = 0; i < node->mbr.size(); i++)
        {
            cout << indent
                 << "[" << node->mbr[i].minX << "," << node->mbr[i].minY
                 << " : " << node->mbr[i].maxX << "," << node->mbr[i].maxY << "]";
            if (node->isLeaf)
            {
                cout << " -> " << node->data[i];
            }
            cout << endl;
            if (!node->isLeaf)
            {
                printRec(node->children[i], depth + 1);
            }
        }
    }

    // ---- 销毁 ----
    void destroyRec(RTreeNode *node)
    {
        if (!node)
            return;
        if (!node->isLeaf)
        {
            for (auto *c : node->children)
                destroyRec(c);
        }
        delete node;
    }

public:
    RTree() : root(nullptr), size_(0) {}

    ~RTree() { destroyRec(root); }

    // 插入一个矩形及其数据
    void insert(int minX, int minY, int maxX, int maxY, int data)
    {
        root = insertRect(root, Rect(minX, minY, maxX, maxY), data);
    }

    void insert(const Rect &r, int data)
    {
        root = insertRect(root, r, data);
    }

    // 范围查询: 返回所有与query相交的数据
    vector<int> search(const Rect &query)
    {
        vector<int> result;
        searchRec(root, query, result);
        return result;
    }

    // 点查询: 返回包含该点的所有数据
    vector<int> searchPoint(int x, int y)
    {
        return search(Rect(x, y, x, y));
    }

    // 删除
    bool remove(const Rect &r, int val)
    {
        if (!root)
            return false;
        bool ok = deleteRec(root, r, val);
        if (root && !root->isLeaf && root->children.size() == 1)
        {
            RTreeNode *newRoot = root->children[0];
            root->children.clear();
            root->mbr.clear();
            root->data.clear();
            delete root;
            root = newRoot;
        }
        if (root && root->isLeaf && root->mbr.empty())
        {
            delete root;
            root = nullptr;
        }
        return ok;
    }

    int size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void print()
    {
        if (!root)
        {
            cout << "R-tree is empty." << endl;
            return;
        }
        cout << "=== R-Tree (size=" << size_ << ") ===" << endl;
        printRec(root, 0);
    }
};

int main()
{
    RTree tree;

    // 插入测试数据
    cout << "=== 插入测试数据 ===" << endl;
    tree.insert(0, 0, 10, 10, 1);
    tree.insert(5, 5, 15, 15, 2);
    tree.insert(20, 20, 30, 30, 3);
    tree.insert(8, 2, 12, 8, 4);
    tree.insert(25, 25, 28, 28, 5);
    tree.insert(2, 20, 8, 28, 6);
    tree.insert(15, 10, 25, 20, 7);
    tree.insert(0, 0, 2, 2, 8);
    tree.insert(5, 5, 6, 6, 9);
    tree.print();

    // 范围查询
    cout << "\n=== 范围查询 [0,0,12,12] ===" << endl;
    vector<int> res = tree.search(Rect(0, 0, 12, 12));
    cout << "找到 " << res.size() << " 条数据: ";
    for (int v : res)
        cout << v << " ";
    cout << endl;

    // 点查询
    cout << "\n=== 点查询 (6,6) ===" << endl;
    res = tree.searchPoint(6, 6);
    cout << "找到 " << res.size() << " 条数据: ";
    for (int v : res)
        cout << v << " ";
    cout << endl;

    // 大数据范围查询
    cout << "\n=== 范围查询 [0,0,50,50] ===" << endl;
    res = tree.search(Rect(0, 0, 50, 50));
    cout << "找到 " << res.size() << " 条数据: ";
    for (int v : res)
        cout << v << " ";
    cout << endl;

    // 删除
    cout << "\n=== 删除数据1 [0,0,10,10] ===" << endl;
    tree.remove(Rect(0, 0, 10, 10), 1);
    tree.print();

    cout << "\n=== 删除数据8 [0,0,2,2] ===" << endl;
    tree.remove(Rect(0, 0, 2, 2), 8);
    tree.print();

    return 0;
}
