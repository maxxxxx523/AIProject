#include <bits/stdc++.h>
using namespace std;

struct TreeNode
{
    int val;
    int height;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), height(1), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), height(1), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *l, TreeNode *r) : val(x), height(1), left(l), right(r) {}
};

class AVLTree
{
private:
    TreeNode *root;

    // 获取节点高度
    int getHeight(TreeNode *node)
    {
        return node ? node->height : 0;
    }

    // 更新节点高度
    void updateHeight(TreeNode *node)
    {
        if (node)
        {
            node->height = max(getHeight(node->left), getHeight(node->right)) + 1;
        }
    }

    // 获取平衡因子
    int getBalanceFactor(TreeNode *node)
    {
        return node ? getHeight(node->left) - getHeight(node->right) : 0;
    }

    // 右旋
    TreeNode *rightRotate(TreeNode *root)
    {
        TreeNode *p = root->left;
        root->left = p->right;
        p->right = root;

        // 更新高度
        updateHeight(root);
        updateHeight(p);

        return p;
    }

    // 左旋
    TreeNode *leftRotate(TreeNode *root)
    {
        TreeNode *p = root->right;
        root->right = p->left;
        p->left = root;

        // 更新高度
        updateHeight(root);
        updateHeight(p);

        return p;
    }


    // 平衡节点
    TreeNode *balance(TreeNode *node)
    {
        if (!node)
            return nullptr;

        // 更新高度
        updateHeight(node);

        // 计算平衡因子
        int balanceFactor = getBalanceFactor(node);

        // LL情况：右旋
        if (balanceFactor > 1 && getBalanceFactor(node->left) >= 0)
        {
            return rightRotate(node);
        }

        // LR情况：先左旋再右旋
        if (balanceFactor > 1 && getBalanceFactor(node->left) < 0)
        {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }

        // RR情况：左旋
        if (balanceFactor < -1 && getBalanceFactor(node->right) <= 0)
        {
            return leftRotate(node);
        }

        // RL情况：先右旋再左旋
        if (balanceFactor < -1 && getBalanceFactor(node->right) > 0)
        {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }

        return node;
    }

    // 插入节点（内部实现）
    TreeNode *insertNode(TreeNode *node, int val)
    {
        if (!node)
        {
            return new TreeNode(val);
        }

        if (val < node->val)
        {
            node->left = insertNode(node->left, val);
        }
        else if (val > node->val)
        {
            node->right = insertNode(node->right, val);
        }
        else
        {
            // 元素已存在，不插入（可根据需要修改为计数模式）
            return node;
        }

        // 平衡并返回
        return balance(node);
    }

    // 找到最小值节点
    TreeNode *findMin(TreeNode *node)
    {
        while (node && node->left)
        {
            node = node->left;
        }
        return node;
    }

    // 删除节点（内部实现）
    TreeNode *deleteNode(TreeNode *node, int val)
    {
        if (!node)
            return nullptr;

        if (val < node->val)
        {
            node->left = deleteNode(node->left, val);
        }
        else if (val > node->val)
        {
            node->right = deleteNode(node->right, val);
        }
        else
        {
            // 找到要删除的节点
            if (!node->left || !node->right)
            {
                // 单子节点或叶子节点
                TreeNode *temp = node->left ? node->left : node->right;
                delete node;
                return temp;
            }
            else
            {
                // 双子节点：找后继节点
                TreeNode *successor = findMin(node->right);
                node->val = successor->val;
                node->right = deleteNode(node->right, successor->val);
            }
        }

        // 平衡并返回
        return balance(node);
    }

    // 中序遍历
    void inorder(TreeNode *node)
    {
        if (node)
        {
            inorder(node->left);
            cout << node->val << " ";
            inorder(node->right);
        }
    }

    // 前序遍历
    void preorder(TreeNode *node)
    {
        if (node)
        {
            cout << node->val << " ";
            preorder(node->left);
            preorder(node->right);
        }
    }

    // 层序遍历（打印树结构）
    void levelOrder(TreeNode *node)
    {
        if (!node)
            return;

        queue<TreeNode *> q;
        q.push(node);

        while (!q.empty())
        {
            int levelSize = q.size();
            for (int i = 0; i < levelSize; i++)
            {
                TreeNode *curr = q.front();
                q.pop();

                cout << curr->val << "(" << curr->height << ") ";

                if (curr->left)
                    q.push(curr->left);
                if (curr->right)
                    q.push(curr->right);
            }
            cout << endl;
        }
    }

    // 查找节点
    bool searchNode(TreeNode *node, int val)
    {
        if (!node)
            return false;

        if (val < node->val)
        {
            return searchNode(node->left, val);
        }
        else if (val > node->val)
        {
            return searchNode(node->right, val);
        }
        else
        {
            return true;
        }
    }

    // 销毁树
    void destroyTree(TreeNode *node)
    {
        if (node)
        {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }

public:
    AVLTree() : root(nullptr) {}

    ~AVLTree()
    {
        destroyTree(root);
    }

    // 插入元素
    void insert(int val)
    {
        root = insertNode(root, val);
    }

    // 删除元素
    void remove(int val)
    {
        root = deleteNode(root, val);
    }

    // 查找元素
    bool search(int val)
    {
        return searchNode(root, val);
    }

    // 中序遍历（输出升序序列）
    void printInorder()
    {
        inorder(root);
        cout << endl;
    }

    // 前序遍历
    void printPreorder()
    {
        preorder(root);
        cout << endl;
    }

    // 打印树结构（层序遍历，显示高度）
    void printTree()
    {
        if (!root)
        {
            cout << "Tree is empty" << endl;
            return;
        }
        levelOrder(root);
    }

    // 判断是否为空
    bool isEmpty()
    {
        return root == nullptr;
    }
};

int main()
{
    AVLTree avl;
    
    // 插入测试数据
    cout << "插入序列: 10, 20, 30, 40, 50, 25" << endl;
    int arr[] = {10, 20, 30, 40, 50, 25};
    for (int val : arr) {
        avl.insert(val);
    }
    
    cout << "\n树结构（层序遍历，括号内为高度）:" << endl;
    avl.printTree();
    
    cout << "\n中序遍历（升序）: ";
    avl.printInorder();
    
    cout << "前序遍历: ";
    avl.printPreorder();
    
    cout << "\n查找元素:" << endl;
    cout << "查找25: " << (avl.search(25) ? "找到" : "未找到") << endl;
    cout << "查找100: " << (avl.search(100) ? "找到" : "未找到") << endl;
    
    cout << "\n删除元素30" << endl;
    avl.remove(30);
    cout << "删除后的中序遍历: ";
    avl.printInorder();
    
    cout << "\n删除元素10" << endl;
    avl.remove(10);
    cout << "删除后的中序遍历: ";
    avl.printInorder();
    system("pause");
    return 0;
}
