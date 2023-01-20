```c++
class Solution {
    bool vis[900][900];
    string path;
public:
    bool hasPath(vector<vector<char>>& matrix, string &str) {
        for (int i = 0; i < matrix.size(); ++i)
            for (int j = 0; j < matrix[0].size(); ++j) {
                memset(vis, 0, sizeof vis);
                vis[i][j] = true;
                path.clear();
                if (dfs(i, j, matrix, str)) return true;
            }
        return false;
    }
    
    // 考察(x,y)
    bool dfs(int x, int y, vector<vector<char>>& matrix, string &str) {
        printf("%d,%d\n", x, y);
        if (matrix[x][y] != str[path.size()]) return false;
        path += matrix[x][y];
        cout << x << " " << y << " " << path << endl;
        if (path.size() >= str.size()) return true;
        
        const int dx[] = {0, 0, -1, 1}, dy[] = {1, -1, 0, 0};
        for (int d = 0;  d < 4; ++d) {
            int nx = x + dx[d], ny = y + dy[d];
            if (nx >= 0 && nx < matrix.size() && ny >= 0 && ny < matrix[0].size() && !vis[nx][ny]) {
                string s = path;
                vis[nx][ny] = true;
                if (dfs(nx, ny, matrix, str)) return true;
                vis[nx][ny] = false;
                path = s;
            }
        }
        
        return false;
    }
};
```

