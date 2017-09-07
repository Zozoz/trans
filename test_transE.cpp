#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
using namespace std;

bool debug=false;
bool L1_flag=1;

string trainortest = "test";
string inPath = "./FB15k/";
string outPath = "./embeddings/";
int n= 50;

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%10==9)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,entity_vec;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    }
    void run()
    {
        int tmp;
        FILE* f1 = fopen((outPath + "rel_embeddings.txt").c_str(),"r");
        FILE* f3 = fopen((outPath + "ent_embeddings.txt").c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        tmp = fscanf(f1, "%d", &tmp);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                tmp = fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        cout << "load relation embeddings done!\n";
        entity_vec.resize(entity_num);
        tmp = fscanf(f3, "%d", &tmp);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                tmp = fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        cout << "load entity embeddings done!\n";
        fclose(f1);
        fclose(f3);
        
        test_relation();
        test_entity();
    }

    void test_relation() {
		double rsum = 0,rsum_filter=0;
		double rp_n=0,rp_n_filter;

        for (int testid = 0; testid<fb_l.size(); testid+=1)
		{
            if (testid % 1000 == 0) cout << testid << endl;
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			double tmp = calc_sum(h,l,rel);
			vector<pair<int,double> > a;
			for (int i=0; i<relation_num; i++)
			{
				double sum = calc_sum(h,l,i);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			int filter = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
			    if (ok[make_pair(h, a[i].first)].count(l)==0)
			    	filter+=1;
				if (a[i].first == rel)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					if (a.size()-i<=1)
					{
						rp_n+=1;
					}
					if (filter<1)
					{
						rp_n_filter+=1;
					}
					break;
				}
			}
        }
        cout << "-------------------------------------------------------------\n";
        cout << "Entity Prediction:\n";
        cout << "mean rank(raw) = " << rsum / fb_l.size() << "\tmean rank(filter) = " << rsum_filter / fb_l.size() << endl;
        cout << "hit@1(raw) = " << rp_n / fb_l.size() << "\thit@1(filter) = " << rp_n_filter / fb_l.size() << endl;
        cout << "-------------------------------------------------------------\n";
    }

    void test_entity() {
		double lsum = 0, lsum_filter = 0;
		double rsum = 0, rsum_filter = 0;
		double lp_n = 0, lp_n_filter = 0;
		double rp_n = 0, rp_n_filter = 0;

        for (int testid = 0; testid<fb_l.size(); testid+=1)
		{
            if (testid % 1000 == 0) cout << testid << endl;
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			double tmp = calc_sum(h,l,rel);
			vector<pair<int,double> > a;
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(i,l,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			int filter = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
			    if (ok[make_pair(a[i].first,rel)].count(l)==0)
			    	filter+=1;
				if (a[i].first ==h)
				{
					lsum+=a.size()-i;
					lsum_filter+=filter+1;
					if (a.size()-i<=10)
					{
						lp_n+=1;
					}
					if (filter<10)
					{
						lp_n_filter+=1;
					}
					break;
				}
			}
			a.clear();
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(h,i,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
			    if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					if (a.size()-i<=10)
					{
						rp_n+=1;
					}
					if (filter<10)
					{
						rp_n_filter+=1;
					}
					break;
				}
			}
        }
        cout << "-------------------------------------------------------------\n";
        cout << "Entity Prediction:\n";
        cout << "left:\n";
        cout << "mean rank(raw) = " << lsum / fb_l.size() << "\tmean rank(filter) = " << lsum_filter / fb_l.size() << endl;
        cout << "hit@10(raw) = " << lp_n / fb_l.size() << "\thit@10(filter) = " << lp_n_filter / fb_l.size() << endl;
        cout << "right:\n";
        cout << "mean rank(raw) = " << rsum / fb_l.size() << "\tmean rank(filter) = " << rsum_filter / fb_l.size() << endl;
        cout << "hit@10(raw) = " << rp_n / fb_l.size() << "\thit@10(filter) = " << rp_n_filter / fb_l.size() << endl;
        cout << "overall:\n";
        cout << "mean rank(raw) = " << (lsum + rsum) / 2.0 / fb_l.size() << "\tmean rank(filter) = " << (lsum_filter + rsum_filter) / 2.0 / fb_l.size() << endl;
        cout << "hit@10(raw) = " << (lp_n + rp_n) / 2.0 / fb_l.size() << "\thit@10(filter) = " << (lp_n_filter + rp_n_filter) / 2.0 / fb_l.size() << endl;
        cout << "-------------------------------------------------------------\n";
    }

};
Test test;

void load_data(string f, bool is_test=false){
    FILE* f_kb = fopen(f.c_str(),"r");
    int tmp, num;
    tmp = fscanf(f_kb, "%d", &num);
    cout << f << " " << num << endl;
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        tmp = fscanf(f_kb,"%s",buf);
        string s2=buf;
        tmp = fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
        	cout<<"miss relation:"<<s3<<endl;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],is_test);
    }
    fclose(f_kb);
}

void prepare()
{
    FILE* f1 = fopen((inPath + "entity2id.txt").c_str(),"r");
	FILE* f2 = fopen((inPath + "relation2id.txt").c_str(),"r");
    int tmp;
	int x;
    tmp = fscanf(f1, "%d", &entity_num);
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
	}
    tmp = fscanf(f2, "%d", &relation_num);
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
	}

    load_data((inPath + "train.txt"), false);
    load_data((inPath + "test.txt"), true);
    load_data((inPath + "valid.txt"), false);
}


int main(int argc,char**argv)
{
    prepare();
    test.run();
}

