#include<bits/stdc++.h>
using namespace std;

#define ll long long int
#define ld long double
#define pb push_back
#define all(X) X.begin(),X.end()

const int level_threshold = 800;
const int n_leaf = 10;

vector < vector < pair < int,double > > > X;
vector < vector < int > > Y;

// assumed each of the data point vectors and label vectors sorted

bool test_leaf(vector<int> instances,int level)
{
	// max depth reached
	if(level == level_threshold)
		return 1;
	
	// cardinality of the node’s instance subset is lower than a given threshold
	int n = instances.size();
	if(n <= n_leaf)
		return 1;

	int i,j;

	// all the instances have the same features
	for(i=1;i<n;++i)
	{
		int cur = instances[i];
		int prv = instances[i-1];
		int d = X[cur].size();
		if(d != (int) X[prv].size())
			break;
		
		for(j=0;j<d;++j)
		{
			if(X[cur][j].first != X[prv][j].first || abs(X[cur][j].second - X[prv][j].second) > 1e-5)
				break;
		}
		if(j < d)
			break;
	}
	if(i == n)
		return 1;

	// all the instances have the same labels
	for(i=1;i<n;++i)
	{
		int cur = instances[i];
		int prv = instances[i-1];
		int d = Y[cur].size();
		if(d != (int) Y[prv].size())
			break;
		
		for(j=0;j<d;++j)
		{
			if(Y[cur][j] != Y[prv][j])
				break;
		}
		if(j < d)
			break;
	}
	if(i == n)
		return 1;
}

int main()
{
	
}

	