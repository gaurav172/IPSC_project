#include<bits/stdc++.h>
using namespace std;

#define ll long long int
#define ld long double
#define pb push_back
#define all(X) X.begin(),X.end()

const int level_threshold = 800;
const int n_leaf = 10;

vector< unordered_map<int,double> > X; // Feature Matrix
vector< unordered_map<int,bool> > Y; // Label matrix
vector< unordered_map<int,double> > mean; // mean vector for storing meanLabels at leaves
vector< vector<int> > child_set; // set of children ids for a given node
vector<int> pos; // reference to positions for a node id in mean vector and classifier vector
vector< vector< unordered_map<int,double> > > classifier; // stpres the classifiers for each node

int N; // total no. of instances
int d_x; // no. of dimensions in data vector
int d_y; // no. of possible labels
int k = 20; // branchinf factor

int NXT_NODE_ID;

// check for leaf condition
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
			if(Y[prv][dim])
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

// computes meanLabel for the leaves
unordered_map<int,double> compute_meanLabel(vector<int> instances)
{
	int n = instances.size();
	unordered_map<int,double> meanLabel;
	for(int i=0;i<n;++i)
	{
		int cur = instances[i];
		for(auto v:Y[cur])
		{
			int dim = v.first;
			meanLabel[dim]++;
		}
	}
	for(auto it = meanLabel.begin();it != meanLabel.end();++it)
		it->second /= n;
	return meanLabel;
}

// 1-cosine_similarity(vector a,vector b)
double similarity(unordered_map<int,double> a,unordered_map<int,double> b)
{
	double ma = 0;
	for(auto v:a)
		ma += v.second*v.second;

	double mb = 0;
	for(auto v:b)
		mb += v.second*v.second;

	double prd = 0;
	for(auto v:a)
		prd += v.second*b[v.first];

	return 1.0-double(prd)/double(sqrt(ma)*sqrt(mb));
}

// Given certain data points in clusters, classify function decides for test point, the most similar data point
int classify(vector< unordered_map<int,double> > node_classifier,unordered_map<int,double> v)
{
	double mx = 0.0;
	int cid = 0;
	for(int i=0;i<(int)node_classifier.size();++i)
	{
		double sim = similarity(v,node_classifier[i]);
		if(sim > mx)
		{
			mx = sim;
			cid = i;
		}
	}
	return cid;
}

// training the node classifier
vector< unordered_map<int,double> > train_node_classifier(vector<int> instances)
{
	
	vector< unordered_map<int,double> > node_classifier;
	return node_classifier;
}

// build function for constructing the tree
void train_tree(vector<int> instances,int level)
{
	int id = NXT_NODE_ID;
	NXT_NODE_ID++;
	if(test_leaf(instances, level))
	{
		// Here pos indicates position of mean label for this node present in mean vector 
		pos.push_back((int) mean.size());
		mean.push_back(compute_meanLabel(instances));
		return;
	}

	// Here pos indicates position of classifier for this node present in classifier vector 
	pos.push_back((int) classifier.size());
	classifier.push_back(train_node_classifier(instances));
	vector< vector<int> > partition(k);
	for(auto v:instances)
	{
		int cid = classify(classifier[pos[id]],X[v]);
		partition[cid].push_back(v);
	}

	vector<int> children;
	for(int i=0;i<k;++i)
	{
		if(!(partition[i].empty()))
		{
			train_tree(partition[i],level+1);
			children.push_back(NXT_NODE_ID-1);
		}
	}

	child_set.push_back(children);
	return;
}

int main()
{
	
}

	