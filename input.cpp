#include <bits/stdc++.h>
using namespace std;

#define PI acos(-1)
#define mp make_pair
#define fi first
#define se second
#define pub push_back
#define pob pop_back
#define forn(i, n) for (int i = 0; i < (int)(n); ++i)
#define fornr(i, n) for (int i = (int)(n) - 1; i >= 0; --i)
#define um unordered_map

using pii = pair<int, int>;
using vi = vector<int>;
using vvi = vector<vector<int>>;
using vpii = vector<pii>;
using ll = long long;
using pll = pair<ll, ll>;
using vll = vector<ll>;
using mii = map<int, int>;
using si = set<int>;
using umii = unordered_map<int, int>;
using umll = unordered_map<ll, ll>;

int main()
{
    ios_base::sync_with_stdio(false), cin.tie(NULL);
    // freopen("input.txt", "r" , "stdin");
    // freopen("output.txt" , "w" , "stdout");

    int n, Q;
    cin >> n >> Q;
    vector<vector<int>> v(n, vector<int>(n, 0));

    while(n--)
    {
        int x , y , q , e, l, s;
        cin >> x >> y >> q >> e >> l >> s;

        v.push_back({x, y, q, e, l, s});
        
    }

    return 0;
}