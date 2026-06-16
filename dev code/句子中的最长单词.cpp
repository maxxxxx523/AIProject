#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
using namespace std;

int main()
{
	string s;
    string ans;
    while (cin >> s) {
        bool end = false;
        // 如果最后一个字符是 '.', 就把它去掉，并标记结束
        int len = s.size();
        if (s[len - 1] == '.') {
            s = s.substr(0, len - 1);
            end = true;
        }
        // 比较长度，更新最长单词
        if (s.size() > ans.size())
            ans = s;
        if (end) break; // 遇到句号，跳出循环
    }
    cout << ans << endl;

	return 0;
}