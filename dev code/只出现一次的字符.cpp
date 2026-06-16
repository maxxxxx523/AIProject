#include <iostream>
#include <string>
#include <unordered_set>
using namespace std;

// 函数功能：在字符串中查找第一个出现奇数次的字符
// 如果所有字符都出现偶数次，返回"no"
string find(const string& str)
{
    unordered_set<char> set;  // 使用集合来记录出现奇数次的字符

    // 遍历字符串中的每个字符
    for (char c : str) {
        // 如果字符已经在集合中，说明这是第二次出现（偶数次），从集合中移除
        if (set.count(c))//统计次数
            set.erase(c);
        // 如果字符不在集合中，说明这是第一次出现（奇数次），添加到集合中
        else
            set.insert(c);
    }

    // 再次遍历字符串，找到第一个出现在集合中的字符
    // 因为集合中保存的都是出现奇数次的字符
    for (size_t i = 0; i < str.length(); i++)
        if (set.count(str[i]))
            return string(1, str[i]);  // 返回该字符的字符串形式

    // 如果集合为空，说明所有字符都出现偶数次
    return "no";
}

int main()
{
    string s;
    cin >> s;           // 从标准输入读取字符串
    cout << find(s);    // 调用find函数并输出结果
    return 0;
}