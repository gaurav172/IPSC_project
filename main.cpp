#include <bits/stdc++.h>
#include "mm.h"
using namespace std;
#define ll long long int
#define umap unordered_map
#define pb push_back
#define ff first
#define ss second
#define sz(a) (ll)a.size()
int n,xdim,ydim,projDimx,projDimy;
vector<umap<int,double>> X,Y,Xp,Yp;
ll seedIndexX,seedSignX,seedIndexY,seedSignY;
vector<int> ProjColX,ProjSignX,ProjColY,ProjSignY;
void getInpX()
{
	string F = "feature_matrix";
	ifstream in(F.c_str());
	string s;
	getline(in,s);
	int c = 0;
	while(c<sz(s))
	{
		if(s[c]>='0' && s[c]<='9')
		{
			n = (n*10+(s[c]-'0'));
			c++;
			continue;
		}
		c++;
		break;
	}
	while(c<sz(s))
	{
		if(s[c]>='0' && s[c]<='9')
		{
			xdim = (xdim*10+(s[c]-'0'));
			c++;
			continue;
		}
		c++;
		break;
	}
	int row = 0;
	while(getline(in,s))
	{
		umap<int,double> vec;
		int c = 0;
		while(c<s.size())
		{
			while(c<sz(s) && s[c]!='<')
				c++;
			if(c==sz(s))
				break;
			c++;
			int col = 0;
			while(s[c]!=':')
			{
				col = (col*10+s[c]-'0');
				c++;
			}
			c++;
			double val = 0;
			while(s[c]!='>')
			{
				val = (val*10+s[c]-'0');
				c++;
			}
			vec[col] = val;
		}
		X.pb(vec);
	}
}
void getInpY()
{
	string L = "label_matrix";
	ifstream in(L.c_str());
	string s;
	getline(in,s);
	int c = 0;
	n = 0;
	while(c<sz(s))
	{
		if(s[c]>='0' && s[c]<='9')
		{
			n = (n*10+(s[c]-'0'));
			c++;
			continue;
		}
		c++;
		break;
	}
	while(c<sz(s))
	{
		if(s[c]>='0' && s[c]<='9')
		{
			ydim = (ydim*10+(s[c]-'0'));
			c++;
			continue;
		}
		c++;
		break;
	}
	int row = 0;
	while(getline(in,s))
	{
		umap<int,double> vec;
		int c = 0;
		while(c<s.size())
		{
			while(c<sz(s) && s[c]!='<')
				c++;
			if(c==sz(s))
				break;
			c++;
			int col = 0;
			while(s[c]!=':')
			{
				col = (col*10+s[c]-'0');
				c++;
			}
			c++;
			double val = 0;
			while(s[c]!='>')
			{
				val = (val*10+s[c]-'0');
				c++;
			}
			vec[col] = val;
		}
		Y.pb(vec);
	}
}
void intialize_seeds()
{
	ll mod = (1LL<<32)-1;
	seedIndexX = (rand()*rand()*rand())%mod;
	seedIndexY = (rand()*rand()*rand())%mod;
	seedSignX = (rand()*rand()*rand())%mod;
	seedSignY = (rand()*rand()*rand())%mod;
}
void ProjectX()
{
	for(int i = 0;i < xdim;i++)
	{
		__int128 p,q;
		const string key = "azv" + to_string(i);
		MurmurHash3_x64_128(&key, (uint64_t)sz(key), seedIndexX, &p);			
		MurmurHash3_x64_128(&key, (uint64_t)sz(key), seedSignX, &q);			
		p = p%(__int128)projDimx;
		q = q%(__int128)2;
		ProjColX.pb((int)p);
		ProjSignX.pb(2*(int)q-1);
	}
	for(int i=0;i<n;i++)
	{
		for(auto u:X[i])
		{
			Xp[i][ProjColX[u.ff]] += (double)ProjSignX[u.ff]*u.ss;
		}
	}
}
void ProjectY()
{
	for(int i = 0;i < ydim;i++)
	{
		__int128 p,q;
		const string key = "azv" + to_string(i);
		MurmurHash3_x64_128(&key, (uint64_t)sz(key), seedIndexY, &p);			
		MurmurHash3_x64_128(&key, (uint64_t)sz(key), seedSignY, &q);			
		p = p%(__int128)projDimy;
		q = q%(__int128)2;
		ProjColY.pb((int)p);
		ProjSignY.pb(2*(int)q-1);
	}
	for(int i=0;i<n;i++)
	{
		for(auto u:Y[i])
		{
			Yp[i][ProjColY[u.ff]] += (double)ProjSignY[u.ff]*u.ss;
		}
	}
}
int main()
{
	getInpX();
	getInpY();
	intialize_seeds();
	ProjectX();
	ProjectY();
}