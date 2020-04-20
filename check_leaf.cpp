#include<bits/stdc++.h>
using namespace std;

#define ll long long int
#define ld long double
#define pb push_back
#define all(X) X.begin(),X.end()

const int level_threshold = 800;
const int n_leaf = 10;

vector< unordered_map<int,double> > X; 
vector< unordered_map<int,bool> > Y;

int N; // total no. of instances
int d_x; // no. of dimensions in data vector
int d_y; // no. of possible labels

int NXT_NODE_ID;

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
		
		bool flag = 1;
		for(auto v:X[cur])
		{
			int dim = v.first;
			double val = v.second;
			if(abs(X[prv][dim]-val) > 1e-5)
			{
				flag = 0;
				break;
			}
		}
		if(!flag)
			break;
	}
	if(i == n) return 1;

	// all the instances have the same labels
	for(i=1;i<n;++i)
	{
		int cur = instances[i];
		int prv = instances[i-1];
		int d = Y[cur].size();
		if(d != (int) Y[prv].size())
			break;
		
		bool flag = 1;
		for(auto v:Y[cur])
		{
			int dim = v.first;
			bool val = v.second;
			if(X[prv][dim] != val)
			{
				flag = 0;
				break;
			}
		}
		if(!flag)
			break;
	}
	if(i == n) return 1;

	return 0;
}

int main()
{
	
}
