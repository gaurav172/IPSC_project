#include <bits/stdc++.h>
using namespace std;
#define all(a) a.begin(), a.end()
#define pb push_back
#define ff first
#define ss second
#define sz(a) (int)a.size()
#define umap unordered_map 
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
vector<int> spherical_kmeans(vector<umap<int,double>> Yp,vector<int> inds,int k)
{
	// inds = Indices which have been chosen to build classifier at current node
	// k = number of clusters
	int n = inds.size();
	vector<umap<int,double>> centroids;
	
	// Store magnitudes of data points 
	vector<double> mags(n,0);
	for(int i=0;i<n;i++)
	{
		int id = inds[i];

		double res = 0;
		for(auto u:Yp[id])
			res = res + u.ss*u.ss;
		
		mags[i] = sqrt(res);
	}

	// Choose the first k points as centroids 
	for(int i=0;i<k;i++)
	{
		int id = inds[i];
		
		umap<int,double> vec;
		if(mags[i]==0)
		{
			vec[0] = 1;
			centroids.pb(vec);
			continue;
		}
		for(auto u:Yp[id])
		{
			vec[u.ff] = u.ss / mags[i];
		}
		centroids.pb(vec);
	}

	vector<int> labels(n,0);
	while(1)
	{
		// Spherical K-means loop
		vector<umap<int,double>> new_centroids;
		for(int i=0;i<k;i++)
		{
			umap<int,double> vec;
			new_centroids.pb(vec);
		}
		
		for(int i=0;i<n;i++)
		{
			int id = inds[i];
			// Find the distance of point from the centroids.
			vector<pair<double,int>> dist;
			for(int j=0;j<k;j++)
			{
				// find dot product of unit vectors of Xp[id] and centroid[j]
				double d = 0;
				for(auto u:Yp[id])
				{
					auto it = centroids[j].find(u.ff);
					if(it!=centroids[j].end() && mags[i])
						d = d + (u.ss*(it->ss))/mags[i];
				}
				dist.pb({1.00-d,j});
			}
			// Assign current point to the closest cluster
			sort(all(dist));
			labels[i] = dist[0].ss;
		}

		// Find new centroids
		for(int i=0;i<n;i++)
		{
			if(mags[i]==0)
				continue;
			int id = inds[i];
			int cent = labels[i];
			for(auto u:Yp[id])
			{
				new_centroids[cent][u.ff] += u.ss/mags[i];
			}
		}
		// Normalize the new centroids.
		for(int j=0;j<k;j++)
		{
			double res = 0;
			for(auto u:new_centroids[j])
				res = res + u.ss*u.ss;
			res = sqrt(res);
			if(res==0)
			{
				res = 1;
				new_centroids[j][0] = 1;
				continue;
			}
			for(auto it = new_centroids[j].begin();it!=new_centroids[j].end();it++)
			{
				it->ss /= res;
			}
		}
		// Check for difference in previous and current centroid.
		bool f = 0;
		double eps = 1e-5;
		for(int i=0;i<k;i++)
		{
			for(auto u:centroids[i])
			{
				double df;
				auto it = new_centroids[i].find(u.ff);
				if(it==new_centroids[i].end())
					df = u.ss;
				else
					df = abs(u.ss - it->ss);
				if(df > eps)
					f=1;
			}
			for(auto u:new_centroids[i])
			{
				double df;
				auto it = centroids[i].find(u.ff);
				if(it == centroids[i].end())
					df = u.ss;
				else
					df = abs(u.ss - it->ss);
				if(df > eps)
					f=1;
			}
		}
		if(f==0)
			break;
	}
	return labels;	
}

vector<umap<int,double>> buildClassifier(vector<umap<int,double>> Xp,vector<int> labels,vector<int> inds,int k,int m)
{
	// Find centroids of X
	vector<umap<int,double>> centroids;
	for(int i=0;i<k;i++)
	{
		umap<int,double> vec;
		centroids.pb(vec);
	}
	int n = inds.size();
	for(int i=0;i<n;i++)
	{
		int id = inds[i];

		double mag = 0;
		for(auto u:Xp[id])
			mag = mag + u.ss*u.ss;
		if(mag==0)
			continue;
		mag = sqrt(mag);
		for(auto u:Xp[id])
			centroids[labels[i]][u.ff] += u.ss/mag;
	}

	for(int i=0;i<k;i++)
	{
		double mag = 0;
		for(auto u:centroids[i])
			mag = mag + u.ss*u.ss;
		mag = sqrt(mag);
		if(mag == 0)
		{
			centroids[i][0] = 1;
			continue;
		}
		for(auto it=centroids[i].begin();it!=centroids[i].end();it++)
			it->ss /= mag;
	}
	return centroids;
}

vector<umap<int,double> > trainNodeClassifier(vector<int> inds,int k)
{
	shuffle(all(inds),rng);	
	int smp = min(sz(inds),100);
	vector<int> samples;
	for(int i=0;i<smp;i++)
		samples.pb(inds[i]);
	// Call spherical K-means to get Labels
	// After getting Lables call buildClassifier to get Classifier.
}

int main()
{

}