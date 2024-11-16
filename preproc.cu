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

#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <list>
#include <stdio.h>
#include "common.h"
#include "cuda_common.h"
using namespace std;

#define DEBUG

__global__ void orderEdges_kernel(int* d_xadj, int* d_adj, int n) { /* erases  -1s from adj list */
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if(u < n) {
		int wp, end, j, p;
		wp = d_xadj[u];
		end = d_xadj[u+1];
		for(p = wp; p < end; p++) {
			j = d_adj[p];
			if(j != -1) {
				d_adj[p] = -1;
				d_adj[wp++] = j;
			}
		}
	}
}

__global__ void degreeSet_kernel(int* d_xadj, int* d_degrees, int n) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if(u < n) {
		d_degrees[u] = d_xadj[u+1] - d_xadj[u];
	}
}

__global__ void degree1_kernel(int* d_xadj, int* d_adj, int* d_tadj, int n, float* d_bc, int* d_weight, bool *d_continue, int* d_degrees/*, int* d_hash*/) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;

	if(u < n) {
		if(d_degrees[u] == 1) { /* degree 1 vertex is found */
			int p, v, end, remwght;
			*d_continue = true;
			d_degrees[u] = 0;
			end = d_xadj[u + 1];
			for(p = d_xadj[u]; p < end; p++) {
				v = d_adj[p];
				if(v != -1) {
					d_adj[p] = -1;
					d_adj[d_tadj[p]] = -1; /* bu satiri basit haliyle yazinca ne kaybediyoruz bakalim */

					remwght = n - d_weight[u];
					d_bc[u] += (d_weight[u] - 1) * remwght;

					atomicAdd(d_bc + v, d_weight[u] * (remwght - 1));
					atomicAdd(d_weight + v, d_weight[u]);
					atomicAdd(d_degrees + v, -1);
					break;
				}
			}
		}
	}
}

void init () {
	int* tmp;
	cudaMalloc((void **)&tmp, sizeof(int));
	cudaFree(tmp);
}

int preprocess(int *xadj, int* adj, int* tadj, int *np, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp) {

	int n = *np;
	int nz = xadj[n];
	fflush(0);

	int *d_xadj, *d_adj, *d_tadj, *d_weight;
	int *d_degrees;
	float *d_bc;
	bool h_continue, *d_continue;
	cudaMalloc((void **)&d_xadj, sizeof(int)*(n+1));
	cudaMalloc((void **)&d_adj, sizeof(int)* nz);
	cudaMalloc((void **)&d_tadj, sizeof(int)* nz);
	cudaMalloc((void **)&d_weight, sizeof(int)* n);
	cudaMalloc((void **)&d_bc, sizeof(float)* n);
	cudaMalloc((void **)&d_degrees, sizeof(int)* n);
	cudaMalloc((void **)&d_continue, sizeof(bool));

	cudaMemcpy(d_xadj, xadj, sizeof(int) * (n+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_adj, adj, sizeof(int) * nz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tadj, tadj, sizeof(int) * nz, cudaMemcpyHostToDevice);
	cudaMemset(d_bc, 0, sizeof(float) * n);
	cudaMemcpy(d_weight, weight, sizeof(int) * n, cudaMemcpyHostToDevice);

	int threads_per_block = n;
	int blocks = 1;
	if(n > MTS){
		blocks = (int)ceil(n / (float)MTS);
		threads_per_block = MTS;
	}
	dim3 grid(blocks);
	dim3 threads(threads_per_block);

	// degree1 removal
	degreeSet_kernel<<<grid,threads>>>(d_xadj, d_degrees, n);
	do{
		h_continue = false;
		cudaMemcpy(d_continue, &h_continue, sizeof(bool), cudaMemcpyHostToDevice);
		degree1_kernel<<<grid,threads>>>(d_xadj, d_adj, d_tadj, n, d_bc, d_weight, d_continue, d_degrees);
		cudaThreadSynchronize();
		CudaCheckError();
		cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);
	} while(h_continue);

	//shrink the pointers and reconstruct xadj and adj
	orderEdges_kernel<<<grid,threads>>>(d_xadj, d_adj, n);

	cudaMemcpy(bc, d_bc, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(weight, d_weight, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(adj, d_adj, sizeof(int) * nz, cudaMemcpyDeviceToHost);

	int i, j;

	cudaFree(d_xadj);
	cudaFree(d_adj);
	cudaFree(d_tadj);
	cudaFree(d_bc);
	cudaFree(d_weight);
	cudaFree(d_continue);

	int ptr = 0, idx = 0;

	for (i = 0; i < n; i++) {
		int flag = 0;
		for (j = xadj[i]; j < xadj[i+1]; j++) {
			if (adj[j] != -1) {
				adj[ptr++] = adj[j];
			}
			else {
				flag = 1;
				xadj[idx++] = ptr;
				break;
			}
		}
		if (!flag)
			xadj[idx++] = ptr;
	}

	for (i = idx; i > 0; i--) {
		xadj[i] = xadj[i-1];
	}
	xadj[0] = 0;

	int vcount;
	for (int i = 0; i < n; i++) {
		if(xadj[i+1] != xadj[i]) {
			bc[vcount] = bc[i];
			weight[vcount] = weight[i];
			map_for_order[i] = vcount;
			reverse_map_for_order[vcount] = i;
			vcount++;
			xadj[vcount] = xadj[i+1];
		}
		else
			fprintf(ofp, "bc[%d]: %lf\n", i, bc[i]);
	}
	for (int i = 0; i < xadj[vcount]; i++) {
		adj[i] = map_for_order[adj[i]];
	}
	*np = vcount;

	return 0;
}

void order_graph (int* xadj, int* adj, int* weight, float* bc, int n, int vcount, int deg1, int* map_for_order, int* reverse_map_for_order) {

	int *new_xadj, *new_adj;

	new_xadj = (int*) calloc((n + 1), sizeof(int));
	new_adj = (int*) malloc(sizeof(int) * xadj[n]);

	int* my_map_for_order = (int *) malloc(n * sizeof(int));
	int* my_reverse_map_for_order = (int *) malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		my_map_for_order[i] = my_reverse_map_for_order[i] = -1;
	}

	int* mark = (int*) calloc((n + 1), sizeof(int));
	int* bfsorder = (int*) malloc((n + 1) * sizeof(int));
	int endofbfsorder = 0;
	int cur = 0;
	int ptr = 0;

	for (int i = 0; i < n; i++) {
		if (xadj[i+1] > xadj[i]) {
			bfsorder[endofbfsorder++] = i;
			mark[i] = 1;
			break;
		}
	}

	while (cur != endofbfsorder) {
		int v = bfsorder[cur];
		my_reverse_map_for_order[cur] = v;
		my_map_for_order[v] = cur;
		for (int j = xadj[v]; j < xadj[v+1]; j++) {
			int w = adj[j];
			if (mark[w] == 0) {
				mark[w] = 1;
				bfsorder[endofbfsorder++] = w;
			}
		}
		cur++;
	}
	for (int i = 0; i < n; i++) {
		if (mark[i] == 0) {
			my_reverse_map_for_order[cur] = i;
			my_map_for_order[i] = cur;
			cur++;
		}
	}


	ptr = 0;
	for (int i = 0; i < n; i++) {
		new_xadj[i+1] = new_xadj[i];
		int u = my_reverse_map_for_order[i];
		for (int j = xadj[u]; j < xadj[u+1]; j++) {
			int val = adj[j];
			if (!(ptr < xadj[n])) {
				printf("ptr is not less than xadj[n]\n");
				exit(1);
			}		
			if (!(val < n)) {
				printf("val is not less than n\n");
				exit(1);
			}
			new_adj[ptr++] = my_map_for_order[val];
			new_xadj[i+1]++;
		}
	}

	free(mark);
	free(bfsorder);

	int* new_weight = (int*) malloc (sizeof(int) * n);
	float* new_bc = (float*) malloc (sizeof(float) * n);
	for (int i = 0; i < n; i++) {
		new_bc[my_map_for_order[i]] = bc[i];
		new_weight[my_map_for_order[i]] = weight[i];
	}


	int* temp_map_for_order = (int *) malloc(vcount * sizeof(int));
	int* temp_reverse_map_for_order = (int *) malloc(vcount * sizeof(int));

	if (deg1) {
		for (int i = 0; i < vcount; i++) {
			if (map_for_order[i] != -1) {
				int u = my_map_for_order[map_for_order[i]];
				temp_map_for_order[i] = u;
				temp_reverse_map_for_order[u] = i;
			}
		}
	}
	else
		for (int i = 0; i < vcount; i++) {
			int u = my_map_for_order[i];
			temp_map_for_order[i] = u;
			temp_reverse_map_for_order[u] = i;
		}

	memcpy(map_for_order, temp_map_for_order, sizeof(int) * vcount);
	memcpy(reverse_map_for_order, temp_reverse_map_for_order, sizeof(int) * vcount);

	free (my_map_for_order);
	free (my_reverse_map_for_order);
	free (temp_map_for_order);
	free (temp_reverse_map_for_order);

	memcpy(xadj, new_xadj, sizeof(int) * (n+1));
	memcpy(adj, new_adj, sizeof(int) * xadj[n]);
	free (new_adj);
	free (new_xadj);

	memcpy(bc, new_bc, sizeof(int)*n);
	memcpy(weight, new_weight, sizeof(int)*n);
	free(new_bc);
	free(new_weight);     

}
