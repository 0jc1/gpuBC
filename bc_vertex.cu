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
#include "common.h"
#include <sys/time.h>
#include <stdio.h>

using namespace std;

__global__ void forward_vertex (int *d_ptrs, int *d_js, int *d_d, int *d_sigma, bool *d_continue, int *d_dist, int n_count) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if(u < n_count){
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u] == *d_dist) {
			int end = d_ptrs[u + 1];
			for(int p = d_ptrs[u]; p < end; p++) {
				int w = d_js[p];
				if(d_d[w] == -1) {
					d_d[w] = *d_dist + 1;
					*d_continue = true;
				}
				if(d_d[w] == *d_dist + 1) {
					atomicAdd(&d_sigma[w], d_sigma[u]);
				}
			}
		}
	}
}

__global__ void backward_vertex (int *d_ptrs, int* d_js, int *d_d, int *d_sigma, float *d_delta, float* d_bc, int *d_dist, int n_count) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if(u < n_count) {
		if(d_d[u] == *d_dist - 1) {
			int end = d_ptrs[u + 1];
			float sum = 0;
			for(int p = d_ptrs[u]; p < end; p++) {
				int w = d_js[p];
				if(d_d[w] == *d_dist) {
					sum += 1.0f*d_sigma[u]/d_sigma[w]*(1.0f+d_delta[w]);
				}
			}
			d_delta[u] += sum;
		}
	}
}

__global__ void backsum_vertex (int s, int *d_d, float *d_delta, float *d_bc, int n_count) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid];
	}
}

__global__ void backsum_vertex_deg1 (int s, int *d_d, float *d_delta, float *d_bc, int n_count, int* d_weight) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid] * d_weight[s];
	}
}

__global__ void init_vertex (int s, int *d_d, int *d_sigma, int n_count, int* d_dist){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n_count) {
		d_d[i] = -1;
		d_sigma[i] = 0;
		if(s == i) {
			d_d[i] = 0;
			d_sigma[i] = 1;
			*d_dist = 0;
		}
	}
}

__global__ void set_int_vertex (int* dest, int val){
	*dest = val;
}

__global__ void init_delta (int *d_weight, float* d_delta, int n_count) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n_count) {
		d_delta[i] = d_weight[i]-1;
	}
}

int bc_vertex (int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc) {

	int *d_ptrs, *d_js, *d_d, *d_sigma,  *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	cudaMalloc((void **)&d_ptrs, sizeof(int) * (n_count + 1));
	cudaMalloc((void **)&d_js, sizeof(int) * e_count);

	cudaMemcpy(d_ptrs, h_ptrs, sizeof(int) * (n_count+1), cudaMemcpyHostToDevice); // xadj array
	cudaMemcpy(d_js, h_js, sizeof(int) * e_count, cudaMemcpyHostToDevice); // adj array

	cudaMalloc((void **)&d_d, sizeof(int) * n_count);

	cudaMalloc((void **)&d_sigma, sizeof(int) * n_count);
	cudaMalloc((void **)&d_delta, sizeof(float) * n_count);
	cudaMalloc((void **)&d_dist, sizeof(int));

	cudaMalloc((void **)&d_bc, sizeof(float) * n_count);
	cudaMemset(d_bc, 0, sizeof(float) * n_count);

	cudaMalloc((void **)&d_continue, sizeof(bool));

	int threads_per_block = n_count;
	int blocks = 1;
	if(n_count > MTS){
		blocks = (int)ceil(n_count/(double)MTS);
		threads_per_block = MTS;
	}

	dim3 grid(blocks);
	dim3 threads(threads_per_block);


#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	for(int i = 0; i < min (nb, n_count); i++) {
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_vertex<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif

		// BFS
		do {
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			cudaMemset(d_continue, 0, sizeof(bool));
			forward_vertex<<<grid,threads>>>(d_ptrs, d_js, d_d, d_sigma, d_continue, d_dist, n_count);
			cudaThreadSynchronize();
			set_int_vertex<<<1,1>>>(d_dist, ++h_dist);
			cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);

#ifdef TIMER
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif
		} while (h_continue);

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		//Back propagation
		cudaMemset(d_delta, 0, sizeof(int) * n_count);
		set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		while (h_dist > 1) {
			backward_vertex<<<grid, threads>>>(d_ptrs, d_js, d_d, d_sigma, d_delta, d_bc, d_dist, n_count);
			cudaThreadSynchronize();
			set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		}
		backsum_vertex<<<grid, threads>>>(i, d_d,  d_delta, d_bc, n_count);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif

	}

	cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost);
	cudaFree(d_ptrs);
	cudaFree(d_js);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);

	return 0;
}

int bc_vertex_deg1 (int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc, int* h_weight) {

	int *d_ptrs, *d_js, *d_d, *d_sigma,  *d_dist, h_dist, *d_weight;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	cudaMalloc((void **)&d_ptrs, sizeof(int) * (n_count + 1));
	cudaMalloc((void **)&d_js, sizeof(int) * e_count);

	cudaMemcpy(d_ptrs, h_ptrs, sizeof(int) * (n_count+1), cudaMemcpyHostToDevice); // xadj array
	cudaMemcpy(d_js, h_js, sizeof(int) * e_count, cudaMemcpyHostToDevice); // adj array

	cudaMalloc((void **)&d_d, sizeof(int) * n_count);

	cudaMalloc((void **)&d_sigma, sizeof(int) * n_count);
	cudaMalloc((void **)&d_delta, sizeof(float) * n_count);
	cudaMalloc((void **)&d_weight, sizeof(int) * n_count);
	cudaMemcpy(d_weight, h_weight, sizeof(int) * n_count, cudaMemcpyHostToDevice); // weight array
	cudaMalloc((void **)&d_dist, sizeof(int));

	cudaMalloc((void **)&d_bc, sizeof(float) * n_count);
	cudaMemcpy(d_bc, h_bc, sizeof(int) * n_count, cudaMemcpyHostToDevice); // bc array

	cudaMalloc((void **)&d_continue, sizeof(bool));

	int threads_per_block = n_count;
	int blocks = 1;
	if(n_count > MTS){
		blocks = (int)ceil(n_count/(double)MTS);
		threads_per_block = MTS;
	}

	dim3 grid(blocks);
	dim3 threads(threads_per_block);


#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	for(int i = 0; i < min (nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_vertex<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif

		// BFS
		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			cudaMemset(d_continue, 0, sizeof(bool));
			forward_vertex<<<grid,threads>>>(d_ptrs, d_js, d_d, d_sigma, d_continue, d_dist, n_count);
			cudaThreadSynchronize();
			set_int_vertex<<<1,1>>>(d_dist, ++h_dist);
			cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);

#ifdef TIMER
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif

		} while(h_continue);

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		//Back propagation

		init_delta<<<grid, threads>>>(d_weight, d_delta, n_count); // deltas are initialized
		set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		while(h_dist > 1) {
			backward_vertex<<<grid, threads>>>(d_ptrs, d_js, d_d, d_sigma, d_delta, d_bc, d_dist, n_count);
			cudaThreadSynchronize();
			set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		}


		backsum_vertex_deg1<<<grid, threads>>>(i, d_d,  d_delta, d_bc, n_count, d_weight);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif

	}

	cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost);
	cudaFree(d_ptrs);
	cudaFree(d_js);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);


	return 0;
}
