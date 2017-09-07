#include <iostream>
#include <cstring>
#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <map>
#include <set>
#include <vector>
#include <time.h>
using namespace std;

const float pi = 3.141592653589793238462643383;

int transeThreads = 24;
int transeTrainTimes = 1000;
int nbatches = 1;
int dimension = 50;
float transeAlpha = 0.001;
float margin = 1;

string inPath = "./FB15k/";
string outPath = "./embeddings/";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};

Triple *trainHead, *trainTail, *trainList;

map<int, set<int> > rel2tail_set, rel2head_set;
map<int, vector<int> > rel2tail_vector, rel2head_vector;
set<int>::iterator it;
map<pair<int,int>, map<int,int> > ok;

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

/*
	There are some math functions for the program initialization.
*/

int rand_new(int max) {
    // srand((unsigned int)(time(NULL)));
    return (unsigned int)rand() % max;
}

unsigned long long *next_random;

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

int norm(float * con) {
    return 0;
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

/*
	Read triples from the training file.
*/

int relationTotal, entityTotal, tripleTotal;
float *relationVec, *entityVec;
float *relationVecDao, *entityVecDao;

void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
    cout << relationTotal << endl;
	fclose(fin);

	relationVec = (float *)calloc(relationTotal * dimension, sizeof(float));
	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	}

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
    cout << entityTotal << endl;
	fclose(fin);

	entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec+i*dimension);
	}

	fin = fopen((inPath + "train.txt.idstyle").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
    cout << tripleTotal << endl;
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	tripleTotal = 0;
    int h, t, r;
	while (fscanf(fin, "%d", &h) == 1) {
		tmp = fscanf(fin, "%d", &t);
		tmp = fscanf(fin, "%d", &r);
        trainList[tripleTotal].h = h;
        trainList[tripleTotal].t = t;
        trainList[tripleTotal].r = r;
		trainHead[tripleTotal].h = h;
		trainHead[tripleTotal].t = t;
		trainHead[tripleTotal].r = r;
		trainTail[tripleTotal].h = h;
		trainTail[tripleTotal].t = t;
		trainTail[tripleTotal].r = r;
		tripleTotal++;

        ok[make_pair(h, r)][t] = 1;
        
        if (rel2tail_set[r].size() == 0 || *(rel2tail_set[r].find(t)) == rel2tail_set[r].size()){
            rel2tail_set[r].insert(t);
            rel2tail_vector[r].push_back(t);
        }
        if (rel2head_set[r].size() == 0 || *(rel2head_set[r].find(h)) == rel2head_set[r].size()){
            rel2head_set[r].insert(h);
            rel2head_vector[r].push_back(h);
        }

	}
    for(map<int, set<int> >::iterator it=rel2tail_set.begin(); it!=rel2tail_set.end(); it++) {
        cout << "-----------------------------\n";
        cout << it->first << endl;
        for (set<int>::iterator tt=(it->second).begin(); tt!=(it->second).end(); tt++) {
            cout << *tt << " ";
        }
        cout << endl;
        cout << "-----------------------------\n";
    }
	fclose(fin);
    cout << "load train done!\n";

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(int)*entityTotal);
	memset(rigTail, -1, sizeof(int)*entityTotal);
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	relationVecDao = (float*)calloc(dimension * relationTotal, sizeof(float));
	entityVecDao = (float*)calloc(dimension * entityTotal, sizeof(float));
}

/*
	Training process of transE.
*/

int transeLen;
int transeBatch;
float res;

float calc_sum(int e1, int e2, int rel) {
	float sum=0;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimension;
        	for (int ii=0; ii < dimension; ii++) {
            		sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);	
            	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimension;
	int lastb1 = e1_b * dimension;
	int lastb2 = e2_b * dimension;
	int lastbr = rel_b * dimension;
	for (int ii=0; ii  < dimension; ii++) {
		float x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
		if (x > 0)
			x = -transeAlpha;
		else
			x = transeAlpha;
		relationVec[lastar + ii] -= x;
		entityVec[lasta1 + ii] -= x;
		entityVec[lasta2 + ii] += x;
		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
		if (x > 0)
			x = transeAlpha;
		else
			x = -transeAlpha;
		relationVec[lastbr + ii] -=  x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a);
	float sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transetrainMode(void *con) {
	int id;
	id = (unsigned long long)(con);
	next_random[id] = rand();
    int h, t, r;
	for (int k = transeBatch / transeThreads; k >= 0; k--) {
		int i = rand_max(id, transeLen);
        h = trainList[i].h;
        t = trainList[i].t;
        r = trainList[i].r;
		int pr = 500;
		int j;
        // int j = rand_new(entityTotal);
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
            /*
            if (rel2tail_vector[r].size() == 1) {
                j = rand_new(entityTotal);
            } else {
                j = rand_new(rel2tail_vector[r].size());
                while (rel2tail_vector[r][j] == t)
                    j = rand_new(rel2tail_vector[r].size());
                j = rel2tail_vector[r][j];
                if (ok[make_pair(h, r)].count(j) > 0) {
                    j = rand_new(entityTotal);
                }
            }
            */
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
            /*
            if (rel2head_vector[r].size() == 1) {
                j = rand_new(entityTotal);
            } else {
                j = rand_new(rel2head_vector[r].size());
                while (rel2head_vector[r][j] == t)
                    j = rand_new(rel2head_vector[r].size());
                j = rel2head_vector[r][j];
                if (ok[make_pair(j, r)].count(t) > 0) {
                    j = rand_new(entityTotal);
                }
            }
            */
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
		}
        // norm
		norm(relationVec + dimension * trainList[i].r);
		norm(entityVec + dimension * trainList[i].h);
		norm(entityVec + dimension * trainList[i].t);
		norm(entityVec + dimension * j);
	}
	pthread_exit(NULL);
}

void* train_transe(void *con) {
	transeLen = tripleTotal;
	transeBatch = transeLen / nbatches;
	next_random = (unsigned long long *)calloc(transeThreads, sizeof(unsigned long long));
	for (int epoch = 0; epoch < transeTrainTimes; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transeThreads * sizeof(pthread_t));
			for (long a = 0; a < transeThreads; a++)
				pthread_create(&pt[a], NULL, transetrainMode,  (void*)a);
			for (long a = 0; a < transeThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of transE.
*/

void out_transe() {
    FILE* f2 = fopen((outPath + "rel_embeddings.txt").c_str(), "w");
	FILE* f3 = fopen((outPath + "ent_embeddings.txt").c_str(), "w");
	fprintf(f2, "%d\n", relationTotal);
	for (int i=0; i < relationTotal; i++) {
		int last = dimension * i;
		for (int ii = 0; ii < dimension; ii++)
			fprintf(f2, "%.6f\t", relationVec[last + ii]);
		fprintf(f2,"\n");
	}
    fprintf(f3, "%d\n", entityTotal);
	for (int  i = 0; i < entityTotal; i++) {
		int last = i * dimension;
		for (int ii = 0; ii < dimension; ii++)
			fprintf(f3, "%.6f\t", entityVec[last + ii] );
		fprintf(f3,"\n");
	}
	fclose(f2);
	fclose(f3);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

/*
	Main function
*/

int main(int argc, char **argv) {
    // int i = 0;
    // if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);
	init();
	train_transe(NULL);
	out_transe();
	return 0;
}
