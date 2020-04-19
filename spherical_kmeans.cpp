#include <bits/stdc++.h>
using namespace std;
#define all(a) a.begin(), a.end()
#define pb push_back

vector<int> spherical_kmeans(vector<vector<double>> data,int k)
{
	int n=data.size();
	vector<int> inds(n,0);
	for(int i=0;i<n;i++)
		inds[i] = i;
	random_shuffle(all(inds));
	double mags[n],cmags[k];
	vector<vector<double> > centroids;
	for(int i=0;i<n;i++)
	{
		double res = 0;
		for(int j=0;j<data[i].size();j++)
		{
			res = res + data[i][j]*data[i][j];
		}
		mags[i] = sqrt(res);
	}
	for(int i=0;i<k;i++)
	{
		centroids.pb(data[inds[i]]);
		cmags[i] = mags[inds[i]];
	}
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
			vector<pair<double,int> > dist;
			for(int j=0;j<k;j++)
			{
				double d = 0;
				for(int p=0;p<data[i].size();p++)
				{
					d = d+data[i][p]*centroids[j][p];
				}
				d/=mags[i]*cmags[j];
				dist.pb({1.00-d,j});
			}
			sort(all(dist));
			int label = dist[0].second;
			for(int p=0;p<data[i].size();p++)
			{
				cents[label][p] += data[i][p];
			}
			cnt[label]++;
			labels[i] = label;
		}
		for(int i=0;i<k;i++)
		{
			double z = cnt[i];
			double res = 0;
			for(int j=0;j<cents[i].size();j++)
			{
				cents[i][j]/=z;
				res += cents[i][j]*cents[i][j];
			}
			cmags[i] = sqrt(res);
		}
		bool f=0;
		double eps = 1e-5;
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

int main()
{

}