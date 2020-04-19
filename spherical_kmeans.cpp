#include <bits/stdc++.h>
using namespace std;
#define all(a) a.begin(), a.end()
#define pb push_back

vector<int> spherical_kmeans(vector<vector<double>> data,int k)
{
	int n=data.size();

	// Randomly shuffle the data
	vector<int> inds(n,0);
	for(int i=0;i<n;i++)
		inds[i] = i;
	random_shuffle(all(inds));
	vector<vector<double> > centroids;
	// Normalize the data as we need cosine similarity in Spherical Kmeans
	for(int i=0;i<n;i++)
	{
		double res = 0;
		for(int j=0;j<data[i].size();j++)
		{
			res = res + data[i][j]*data[i][j];
		}
		res = sqrt(res);
		if(res==0)
			continue;
		for(int j=0;j<data[i].size();j++)
			data[i][j]/=res;
	}
	// Choose first k points as the centroids.
	for(int i=0;i<k;i++)
		centroids.pb(data[inds[i]]);

	vector<int> labels(n,0);
	while(1)
	{
		vector<int> cnt(k,0);
		vector<vector<double> > cents;
		for(int i=0;i<k;i++)
		{
			vector<double> p(data[0].size(),0);
			cents.pb(p);
		}
		for(int id=0;id<n;id++)
		{
			int i=inds[id];
			// Find distance of point from the other centroids
			vector<pair<double,int> > dist;
			for(int j=0;j<k;j++)
			{
				double d = 0;
				for(int p=0;p<data[i].size();p++)
				{
					d = d+data[i][p]*centroids[j][p];
				}
				dist.pb({1.00-d,j});
			}
			sort(all(dist));
			// Choose the one closest to it.
			int label = dist[0].second;
			// update the new centroid
			for(int p=0;p<data[i].size();p++)
			{
				cents[label][p] += data[i][p];
			}
			labels[i] = label;
		}
		// Normalize the new centroid
		for(int i=0;i<k;i++)
		{
			double res = 0;
			for(int j=0;j<cents[i].size();j++)
			{
				res = res + cents[i][j]*cents[i][j];
			}
			res = sqrt(res);
			if(res==0)
				continue;
			for(int j=0;j<cents[i].size();j++)
				cents[i][j]/=res;
		}
		bool f=0;
		double eps = 1e-5;
		// Check if no change after the previous round.
		for(int i=0;i<k;i++)
		{
			for(int j=0;j<cents[i].size();j++)
			{
				if(abs(cents[i][j]-centroids[i][j])>eps)
					f=1;
			}
		}
		centroids = cents;
		if(f==0)
			break;
	}
	return labels;
}
vector<vector<double>> buildClassifier(vector<vector<double>> data,vector<int> labels,int k)
{
	vector<vector<double>> centroids;
	int n = data.size();
	for(int i=0;i<k;i++)
	{
		vector<double> vec(data[0].size(),0);
		centroids.pb(vec);
	}
	// Normalzie the Data
	for(int i=0;i<n;i++)
	{
		double res = 0;
		for(int j=0;j<data[i].size();j++)
			res = res + data[i][j]*data[i][j];
		res = sqrt(res);
		if(res==0)
			continue;
		// Compute the centroid
		int lbl = labels[i];
		for(int j=0;j<data[i].size();j++)
			centroids[lbl][j] += data[i][j]/res;
	}
	// Normalize the centroids.
	for(int i=0;i<k;i++)
	{
		double res = 0;
		for(int j=0;j<centroids[i].size();j++)
			res = res + centroids[i][j]*centroids[i][j];
		res = sqrt(res);
		if(res==0)
			continue;
		for(int j=0;j<centroids[i].size();j++)
			centroids[i][j] /= res;
	}
	return centroids;
}
int main()
{
	int n=100,k=5;
	int m=7;
	vector<vector<double>> vec;
	for(int i=0;i<n;i++)
	{
		vector<double> vt;
		for(int j=0;j<m;j++)
			vt.pb(rand()%20);
		vec.pb(vt);
	}
	auto ans = spherical_kmeans(vec,k);
	auto Classifier = buildClassifier(vec,ans,k);
}