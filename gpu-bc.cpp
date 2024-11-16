/*
---------------------------------------------------------------------
 This file is a part of the source code for the paper "Betweenness
 Centrality on GPUs and Heterogeneous Architectures", published in
 GPGPU'13 workshop. If you use the code, please cite the paper.

 Copyright (c) 2013,
 By:    Ahmet Erdem Sariyuce,
        Kamer Kaya,
        Erik Saule,
        Umit V. Catalyurek
---------------------------------------------------------------------
 This file is licensed under the Apache License. For more licensing
 information, please see the README.txt and LICENSE.txt files in the
 main directory.
---------------------------------------------------------------------
*/

#include <vector> 
#include <map>
#include <list>
#include <stdio.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "ulib.h"
#include "timestamp.hpp"
#include "graph.h"
#include <assert.h>

#define DEBUG
#define TIME

#define WARP 16
#define MAXLOAD 8

using namespace std;


void init();
int bc_vertex_deg1 (int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc, int* h_weight);
int bc_edge_deg1(int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc);
int bc_virtual_deg1(int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc, int* weight);
int bc_virtual_deg1_hetero(int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc, int* h_weight, int* xadj, int* adj, int nz, int* weight);
int bc_virtual_coalesced_deg1_hetero(int* h_vmap, int* h_vptrs, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride, int e_count, int virn_count, int nb, float *h_bc, int* h_weight, int* xadj, int* adj, int nz, int* weight);
int bc_vertex(int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc);
int bc_edge(int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc);
int bc_virtual(int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc);
int bc_virtual_coalesced(int* h_vmap, int* h_vptrs, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride, int e_count, int virn_count, int nb, float *h_bc);
int createVirtualCSR(int* ptrs, int* js, int nov, int* vmap, int* virptrs, int maxload, int permuteAdj);
int createVirtualCoalescedCSR(int* ptrs, int* js, int nov, int* vmap, int* virptrs, int* startoffset, int* stride, int maxload, int permuteAdj) ;
void order_graph (int* xadj, int* adj, int* weight, float* bc, int n, int vcount, int deg1, int* map_for_order, int* reverse_map_for_order);
int preprocess(int *xadj, int* adj, int* tadj, int *n, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp);
template <typename VtxType, typename WeightType>
void ReadGraphFromFile(FILE *fpin, VtxType *numofvertex, VtxType **pxadj, VtxType **padjncy, WeightType **padjncyw, WeightType **ppvw);
template <typename VtxType, typename WeightType>
void ReadGraphFromMMFile(FILE *matfp, VtxType *numofvertex, VtxType **pxadj, VtxType **padjncy, WeightType **padjncyw, WeightType **ppvw);
static int really_read(std::istream& is, char* buf, size_t global_size);
template <typename VtxType, typename EdgeType, typename WeightType>
void ReadBinary(char *filename, VtxType *numofvertex_r, VtxType *numofvertex_c, EdgeType **pxadj, VtxType **padj, WeightType **padjw, WeightType **ppvw);
template <typename VtxType, typename WeightType>
void ReadGraph(char *filename, VtxType *numofvertex, VtxType **pxadj, VtxType **padjncy, int** ptadj, WeightType **padjncyw, WeightType **ppvw);


void bc_cpu (int* xadj, int* adj, int nVtx, int nz, int nb, float* bc) {

	for (int i = 0; i < nVtx; i++)
		bc[i] = 0.;

	int* bfsorder = new int[nVtx];
	int* Pred = new int[xadj[nVtx]];
	int* endpred = new int[nVtx];
	int* level = new int[nVtx];
	int* sigma = new int[nVtx];
	float* delta = new float[nVtx];

	for (int source = 0; source < min (nb, nVtx); source++) {
		int endofbfsorder = 1;
		bfsorder[0] = source;

		for (int i = 0; i < nVtx; i++)
			endpred[i] = xadj[i];

		for (int i = 0; i < nVtx; i++)
			level[i] = -2;
		level[source] = 0;

		for (int i = 0; i < nVtx; i++)
			sigma[i] = 0;
		sigma[source] = 1;

		//step 1: build shortest path graph
		int cur = 0;
		while (cur != endofbfsorder) {
			int v = bfsorder[cur];
			for (int j = xadj[v]; j < xadj[v+1]; j++) {
				int w = adj[j];
				if (level[w] < 0) {
					level[w] = level[v]+1;
					bfsorder[endofbfsorder++] = w;
				}
				if (level[w] == level[v]+1) {
					sigma[w] += sigma[v];
					assert (sigma[w] > 0); //check for overflow
					//assert (isfinite(sigma[w]));
				}
				else if (level[w] == level[v] - 1) {
					Pred[endpred[v]++] = w;
				}
			}
			cur++;
		}

		for (int i = 0; i < nVtx; i++) {
			delta[i] = 0.;
		}

		//step 2: compute betweenness
		for (int i = endofbfsorder - 1; i > 0; i--) {
			int w = bfsorder[i];
			for (int j = xadj[w]; j < endpred[w]; j++) {
				int v = Pred[j];
				delta[v] += (sigma[v] * (1 + delta[w])) / sigma[w];
			}
			bc[w] += delta[w];
		}
	}

	delete[] bfsorder;
	delete[] Pred;
	delete[] level;
	delete[] sigma;
	delete[] delta;
	delete[] endpred;
}


void bc_cpu_deg1_one_source (int* xadj, int* adj, int nVtx, int nz, float* bc, int* weight, int source, int* bfsorder, int* Pred, int* endpred, int* level, int* sigma, float* delta ) {
	int endofbfsorder = 1;
	bfsorder[0] = source;

	for (int i = 0; i < nVtx; i++)
		endpred[i] = xadj[i];

	for (int i = 0; i < nVtx; i++)
		level[i] = -2;
	level[source] = 0;

	for (int i = 0; i < nVtx; i++)
		sigma[i] = 0;
	sigma[source] = 1;

	//step 1: build shortest path graph
	int cur = 0;
	while (cur != endofbfsorder) {
		int v = bfsorder[cur];
		for (int j = xadj[v]; j < xadj[v+1]; j++) {
			int w = adj[j];
			if (level[w] < 0) {
				level[w] = level[v]+1;
				bfsorder[endofbfsorder++] = w;
			}
			if (level[w] == level[v]+1) {
				sigma[w] += sigma[v];
			}
			else if (level[w] == level[v] - 1) {
				Pred[endpred[v]++] = w;
			}
		}
		cur++;
	}

	for (int i = 0; i < nVtx; i++) {
		delta[i] = weight[i] - 1;
	}

	//step 2: compute betweenness
	for (int i = endofbfsorder - 1; i > 0; i--) {
		int w = bfsorder[i];
		for (int j = xadj[w]; j < endpred[w]; j++) {
			int v = Pred[j];
			delta[v] += (sigma[v] * (1 + delta[w])) / sigma[w];
		}
		bc[w] += delta[w] * weight[source];
	}

}


void bc_cpu_deg1 (int* xadj, int* adj, int nVtx, int nz, int nb, float* bc, int* weight) {

	int* bfsorder = new int[nVtx];
	int* Pred = new int[xadj[nVtx]];
	int* endpred = new int[nVtx];
	int* level = new int[nVtx];
	int* sigma = new int[nVtx];
	float* delta = new float[nVtx];

	for (int source = 0; source < min (nb, nVtx); source++) {
		bc_cpu_deg1_one_source (xadj, adj, nVtx, nz, bc, weight, source, bfsorder, Pred, endpred, level, sigma, delta);
	}

	delete[] bfsorder;
	delete[] Pred;
	delete[] level;
	delete[] sigma;
	delete[] delta;
	delete[] endpred;
}


int intcmp(const void *v1, const void *v2) {
	return (*(int *)v1 - *(int *)v2);
}


int intcmprev(const void *v1, const void *v2) {
	return (*(int *)v2 - *(int *)v1);
}


void compute_deg_dist (int* xadj, int n) {
	int* degs = (int*) calloc(n, sizeof(int));
	for (int i = 0; i < n; i++) {
		degs[xadj[i+1] - xadj[i]]++;
	}

	for (int i = 0; i < n; i++) {
		if (degs[i] > 0)
			cout<<i<<" "<<degs[i]<<endl;
	}

	free(degs);
	exit(1);
}


int main(int argc, char** argv) {

	char c;
	char* infilename, outfilename;
	int threads_per_block = -1, paropt = -1, nb = 1, xpar = 0, times = 1, sortopt = 0;

	if (argc != 7) {
		std::cout<<"usage: "<<argv[0]<<" <filename> <will_order:0|1> <deg1:0|1> <kernel> <nbsource> <xpar>"<<std::endl;
		return -1;
	}

	init();

	int will_order = atoi(argv[2]);
	int deg1 = atoi(argv[3]);
	int btype = atoi(argv[4]);

	nb = atoi(argv[5]);
	xpar = atoi(argv[6]); // max-deg # of a virtual vertex

	int n, i, j, nVtx;
	int *xadj, *adj, *tadj, *mark, *queue, *new_xadj, *new_adj;

	int count, maxcompid, u, v, p;
	int nocomp;
	int qptr, qeptr, largestSize, compsize, lcompid;

	long* reverse_map = NULL;
	bool do_mapping = ReadGraph<int, int>(argv[1], &n, &xadj, &adj, &tadj, NULL, NULL, &reverse_map);
	nVtx = n;
	int* compid = (int*) malloc(sizeof(int) * n);
	for(i = 0; i < n; i++)
		compid[i] = -1;
	int* que = (int*) malloc(sizeof(int) * n);

	printf("there are %d vertices %d edges\n", n, xadj[n]);

	nocomp = qptr = qeptr = largestSize = 0;
	for (int i = 0; i < n; i++) {

		if(compid[i] == -1) {
			compsize = 1;
			compid[i] = nocomp;
			que[qptr++] = i;

			while(qeptr < qptr) {
				u = que[qeptr++];
				for(p = xadj[u]; p < xadj[u+1]; p++) {
					v = adj[p];
					if(compid[v] == -1) {
						compid[v] = nocomp;
						que[qptr++] = v;
						compsize++;
					}
				}
			}
			if(largestSize < compsize) {
				lcompid = nocomp;
				largestSize = compsize;
			}
			nocomp++;
		}
	}

	int nz = xadj[n];
	int ecount = 0;
	int vcount = 0;

	for(i = 0; i < n; i++) {
		if(compid[i] == lcompid) {
			que[i] = vcount++;
			for(p = xadj[i]; p < xadj[i+1]; p++) {
				if(compid[adj[p]] == lcompid)
					ecount++;
			}
		}
	}

	int* lxadj = (int*) malloc(sizeof(int) * (vcount+1));
	int* ladj = (int*) malloc(sizeof(int) * (ecount));
	int* ltadj = (int*) malloc(sizeof(int) * (ecount));
	vcount = 0; ecount = 0;
	lxadj[0] = 0;
	for(i = 0; i < n; i++) {
		if(compid[i] == lcompid)  {
			vcount++;

			for(p = xadj[i]; p < xadj[i+1]; p++) {
				if(compid[adj[p]] == lcompid) {
					ladj[ecount++] = que[adj[p]];
				}
			}
			lxadj[vcount] = ecount;
		}
	}
	printf("largest component graph obtained with %d vertices %d edges -- %d\n", vcount, ecount, lxadj[vcount]);


	n = vcount;
	nz = ecount;
	free(xadj); xadj = lxadj;
	free(adj); adj = ladj;
	free(tadj); tadj = ltadj;

	//printf("before malloc\n");
	//fflush(0);

	int* degs = (int*)malloc(sizeof(int) * n);
	int* myedges = (int*)malloc(sizeof(int) * nz);
	memcpy(degs, xadj, sizeof(int) * n);

	int ptr;
	for(i = 0; i < n; i++) {
		for(ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
			j = adj[ptr];
			myedges[degs[j]++] = i;
		}
	}

	//printf("after malloc\n");
        //fflush(0);


	for(i = 0; i < n; i++) {
		if(xadj[i+1] != degs[i]) {
			printf("something is wrong i %d xadj[i+1] %d degs[i] %d\n", i, xadj[i+1], degs[i]);
			exit(1);
		}
	}

	memcpy(adj, myedges, sizeof(int) * xadj[n]);
	for(i = 0; i < n; i++) {
		for(ptr = xadj[i]+1; ptr < xadj[i+1]; ptr++) {
			if(adj[ptr] <= adj[ptr-1]) {
				printf("is not sorted\n");
				exit(1);
			}
		}
	}

        //printf("more after malloc\n");
        //fflush(0);

	memcpy(degs, xadj, sizeof(int) * n);
	for(i = 0; i < n; i++) {
		for(ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
			j = adj[ptr];
			if(i < j) {
				tadj[ptr] = degs[j];
				tadj[degs[j]++] = ptr;
			}
		}
	}

	free(degs);
	free(myedges);

        //printf("more more after malloc\n");
        //fflush(0);

	for(i = 0; i < n; i++) {
		for(ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
			j = adj[ptr];
			if((adj[tadj[ptr]] != i) || (tadj[ptr] < xadj[j]) || (tadj[ptr] >= xadj[j+1])) {
				printf("error i %d j %d ptr %d\n", i, j, ptr);
				printf("error  xadj[j] %d  xadj[j+1] %d\n",  xadj[j], xadj[j+1]);
				printf("error tadj[ptr] %d\n", tadj[ptr]);
				printf("error adj[tadj[ptr]] %d\n", adj[tadj[ptr]]);
				exit(1);
			}
		}
	}

        //printf("more more more after malloc\n");
        //fflush(0);

	int* map_for_order = (int *) malloc(n * sizeof(int));
	int* reverse_map_for_order = (int *) malloc(n * sizeof(int));
	int* weight = (int *) malloc(sizeof(int) * n);
	float* bc  = (float *) malloc(sizeof(float) * n);

	for(i = 0; i < n; i++) {
		weight[i] = 1;
		map_for_order[i] = -1;
		reverse_map_for_order[i] = -1;
	}


	struct timeval t1, t2, t3, t4, t5, t6, t7, gt1, gt2;
	t1.tv_sec = t1.tv_usec = t2.tv_sec = t2.tv_usec = t3.tv_sec = t3.tv_usec = t4.tv_sec = t4.tv_usec = t5.tv_sec = t5.tv_usec = t6.tv_sec = t6.tv_usec = t7.tv_sec = t7.tv_usec = 0;
	double time_preproc, time_order, time_kernel, time_total, time_virt;


	FILE* ofp;
	ofp = fopen("bc_out.txt", "w");

	gettimeofday (&t1, 0);
	if (deg1 == 1) {
		preprocess (xadj, adj, tadj, &n, bc, weight, map_for_order, reverse_map_for_order, ofp);
		nz = xadj[n];
	}

	gettimeofday (&t2, 0);

	if (will_order == 1) {
		order_graph (xadj, adj, weight, bc, n, vcount, deg1, map_for_order, reverse_map_for_order);
	}

	gettimeofday (&t3, 0);

	nb = min(n, nb);

	printf("will be executed on %d vertices %d %d edges\n", n, xadj[n], nz); fflush(0);

	// kernels..
	if (deg1 == 1) {
		if (btype == 0) {
			printf("BC: CPU based on compressed graph\n");
			bc_cpu_deg1 (xadj, adj, n, nz, nb, bc, weight);
		}
		else if (btype == 1) {
			printf("BC: vertex based parallelism on compressed graph\n");
			bc_vertex_deg1 (xadj, adj, n, nz, nb, bc, weight);
		}
		else if(btype == 2) {
			printf("there is no such a thing!\n");
			exit(1);
			printf("BC: edge based parallelism on compressed graph\n");

			int* is = (int*) malloc(sizeof(int) * nz);
			for(i = 0; i < n; i++) {
				for(ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
					is[ptr] = i;
				}
			}

			bc_edge_deg1 (is, adj, n, nz, nb, bc);

			free (is);
		}
		else if(btype == 3) {

			printf("BC: virtual vertex based parallelism on compressed graph\n");

			int* extv = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));
			int* start = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));

			gettimeofday (&t5, 0);

			int nov_ext = createVirtualCSR(xadj, adj, n, extv, start, xpar, 0);

			gettimeofday (&t6, 0);
			bc_virtual_deg1_hetero (extv, start, adj, n, nz, nov_ext, nb, bc, weight, xadj, adj, nz, weight);


			free (extv);
			free (start);
		}
		else if(btype == 4) {

			printf("BC: virtual vertex based parallelism with coalesced access on compressed graph\n");

			int* extv = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));
			int* start = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));
			int* startoffset = (int*)malloc(sizeof(int)*(nz + 1 + (xpar * WARP))); //basically xadj[vmap[thread]]+thread_in_vvertex
			int* stride = (int*)malloc(sizeof(int)*(n)); //stride is actually number of virtual vertex per actual vertex

			gettimeofday (&t5, 0);

			int nov_ext = createVirtualCoalescedCSR(xadj, adj, n, extv, start, startoffset, stride, xpar, 0);

			gettimeofday (&t6, 0);

			bc_virtual_coalesced_deg1_hetero (extv, start, xadj, adj, n, startoffset, stride, nz, nov_ext, nb, bc, weight, xadj, adj, nz, weight);
			free (extv);
			free (start);
		}

	}
	else {
		if (btype == 0) {
			printf("BC: CPU based\n");
			bc_cpu (xadj, adj, n, nz, nb, bc);
		}
		else if (btype == 1) {
			printf("BC: vertex based parallelism\n");
			bc_vertex (xadj, adj, n, nz, nb, bc);
		}
		else if(btype == 2) {
			printf("BC: edge based parallelism\n");

			int* is = (int*) malloc(sizeof(int) * nz);
			for(i = 0; i < n; i++) {
				for(ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
					is[ptr] = i;
				}
			}

			bc_edge (is, adj, n, nz, nb, bc);

			free (is);
		}
		else if(btype == 3) {

			printf("BC: virtual vertex based parallelism\n");

			int* extv = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));
			int* start = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));

			gettimeofday (&t5, 0);

			int nov_ext = createVirtualCSR(xadj, adj, n, extv, start, xpar, 0);

			gettimeofday (&t6, 0);

			bc_virtual (extv, start, adj, n, nz, nov_ext, nb, bc);
			free (extv);
			free (start);
		}
		else if(btype == 4) {

			printf("BC: virtual vertex based parallelism with coalesced access\n");

			int* extv = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));
			int* start = (int*)malloc(sizeof(int) * (nz + 1 + (xpar * WARP)));
			int* startoffset = (int*)malloc(sizeof(int)*(nz + 1 + (xpar * WARP))); //basically xadj[vmap[thread]]+thread_in_vvertex
			int* stride = (int*)malloc(sizeof(int)*(n)); //stride is actually number of virtual vertex per actual vertex

			gettimeofday (&t5, 0);

			int nov_ext = createVirtualCoalescedCSR(xadj, adj, n, extv, start, startoffset, stride, xpar, 0);

			gettimeofday (&t6, 0);

			bc_virtual_coalesced (extv, start, xadj, adj, n, startoffset, stride, nz, nov_ext, nb, bc);
			free (extv);
			free (start);
		}


	}

	gettimeofday (&t4, 0);


	if (will_order+deg1 > 0)
		for (int i = 0; i < n; i++) {
			if (reverse_map_for_order[i] != -1)
				fprintf(ofp, "bc[%d]: %lf\n", reverse_map_for_order[i], bc[i]);
		}
	else
		for (int i = 0; i < n; i++) {
				fprintf(ofp, "bc[%d]: %lf\n", i, bc[i]);
		}

	fclose(ofp);
	FILE* lfp;
	lfp = fopen("bc_out.txt", "r");
	char a,b,d,e,f,g, h;
	double val;
	int id;
	double* result_bc = new double[nVtx];

	while (fscanf(lfp, "%c%c%c%d%c%c%c%lf%c", &a, &b, &f, &id, &d, &e, &g, &val, &h) != EOF) {
	  result_bc[id] = val;
	}
	fclose(lfp);
	ofp = fopen("bc_out.txt", "w");
	if (do_mapping == false) {
		for(int i = 0; i < nVtx; i++)
			fprintf(ofp, "bc[%d]: %lf\n", i, result_bc[i]/2);
	}
	else {
		double* last_bc = new double[nVtx];
		for(int i = 0; i < nVtx; i++)
			last_bc[reverse_map[i]] = result_bc[i]/2;
		for(int i = 0; i < nVtx; i++)
			fprintf(ofp, "bc[%d]: %lf\n", i, last_bc[i]);
	}
	fclose(ofp);
	delete[] result_bc;

	free (bc);



	time_preproc = (1000000.0 * (t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec) / 1000000.0;
	cout << "preproc time: " <<time_preproc<<" secs\n";
	time_order = (1000000.0 * (t3.tv_sec-t2.tv_sec) + t3.tv_usec-t2.tv_usec) / 1000000.0;
	cout << "ordering time: " <<time_order<<" secs\n";
	time_kernel = (1000000.0 * (t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec) / 1000000.0;
	cout << "kernel time: " <<time_kernel<<" secs\n";
	time_total = (1000000.0 * (t4.tv_sec-t1.tv_sec) + t4.tv_usec-t1.tv_usec) / 1000000.0;
	cout << "total time: " <<time_total<<" secs "<<"for "<<nb<<" bfs calls\n";
	time_virt = (1000000.0 * (t6.tv_sec-t5.tv_sec) + t6.tv_usec-t5.tv_usec) / 1000000.0;
	cout << "virtualization time: " <<time_virt<<" secs "<<"for "<<nb<<" bfs calls\n";


	return 0;
}

