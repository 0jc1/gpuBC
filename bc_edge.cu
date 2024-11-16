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
#include "cuda_common.h"
#include <assert.h>

using namespace std;

__global__ void forward_edge (int *d_v, int *d_e, int  *d_d, int *d_sigma, bool *d_continue, int *d_dist, int e_count) {

	int tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(tid < e_count) {
		/* for each edge (u, w) */
		int u = d_v[tid];
		if(d_d[u] == *d_dist) {
			int w = d_e[tid];
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

__global__ void backward_edge (int *d_v, int *d_e, int *d_d, int *d_sigma, float *d_delta, int *d_dist, int e_count) {

	int tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(tid < e_count) {
		int u = d_v[tid];
		if(d_d[u] == *d_dist - 1) {
			int w = d_e[tid];
			if(d_d[w] == *d_dist) {
				atomicAdd(&d_delta[u], 1.0f*d_sigma[u]/d_sigma[w]*(1.0f+d_delta[w]));
			}
		}
	}
}

__global__ void backsum_edge (int s, int *d_d, float *d_delta, float *d_bc, int n_count) {

	int tid =  blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid];
	}
}

__global__ void init_edge (int s, int *d_d, int *d_sigma, int n_count, int* d_dist) {

	int i =  blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
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

__global__ void set_int_edge (int* dest, int val) {
	*dest = val;
}

int bc_edge (int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc) {
	int *d_v, *d_e, *d_d, *d_sigma, *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	assert (cudaSuccess == cudaMalloc((void **)&d_v, sizeof(int)*e_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_e, sizeof(int)*e_count));

	assert (cudaSuccess == cudaMemcpy(d_v, h_v, sizeof(int)*e_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_e, h_e, sizeof(int)*e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(int)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(int)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	int threads_per_block = e_count;
	int blocks = 1;
	if(e_count > MTS) {
		blocks = (int)ceil(e_count/(float)MTS);
		blocks = (int)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);
	int threads_per_block2=n_count;
	int blocks2 = 1;
	if(n_count > MTS) {
		blocks2 = (int)ceil(n_count/(double)MTS);
		blocks2 = (int)ceil(sqrt((float)blocks2));
		threads_per_block2 = MTS;
	}
	dim3 grid2;
	grid2.x = blocks2;
	grid2.y = blocks2;
	dim3 threads2(threads_per_block2);


	cout<<"cuda parameters: "<<blocks<<" "<<threads_per_block<<" "<<blocks2<<" "<<threads_per_block2<<endl;

#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	for(int i = 0; i < min(nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_edge <<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

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

			assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
			forward_edge <<<grid,threads>>>(d_v, d_e, d_d, d_sigma, d_continue, d_dist, e_count);
			cudaThreadSynchronize();
			set_int_edge <<<1,1>>>(d_dist, ++h_dist);
			CudaCheckError();
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

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
		assert (cudaSuccess == cudaMemset(d_delta, 0, sizeof(int) * n_count));
		set_int_edge <<<1,1>>>(d_dist, --h_dist);
		while(h_dist > 1) {
			backward_edge <<<grid, threads>>>(d_v, d_e, d_d, d_sigma, d_delta, d_dist, e_count);
			cudaThreadSynchronize();
			set_int_edge <<<1,1>>>(d_dist, --h_dist);
			CudaCheckError();
		}
		backsum_edge <<<grid2, threads2>>>(i, d_d,  d_delta, d_bc, n_count);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_v);
	cudaFree(d_e);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	return 0;
}

int bc_edge_deg1 (int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc) {
	int *d_v, *d_e, *d_d, *d_sigma, *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	assert (cudaSuccess == cudaMalloc((void **)&d_v, sizeof(int)*e_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_e, sizeof(int)*e_count));

	assert (cudaSuccess == cudaMemcpy(d_v, h_v, sizeof(int)*e_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_e, h_e, sizeof(int)*e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(int)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(int)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	int threads_per_block = e_count;
	int blocks = 1;
	if(e_count > MTS) {
		blocks = (int)ceil(e_count/(float)MTS);
		threads_per_block = MTS;
	}
	dim3 grid(blocks);
	dim3 threads(threads_per_block);
	int threads_per_block2=n_count;
	int blocks2 = 1;
	if(n_count > MTS){
		blocks2 = (int)ceil(n_count/(double)MTS);
		threads_per_block2 = MTS;
	}
	dim3 grid2(blocks2);
	dim3 threads2(threads_per_block2);


#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif
	for(int i = 0; i < min(nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_edge<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

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

			assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
			forward_edge<<<grid,threads>>>(d_v, d_e, d_d, d_sigma, d_continue, d_dist, e_count);
			cudaThreadSynchronize();
			set_int_edge<<<1,1>>>(d_dist, ++h_dist);
			CudaCheckError();
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

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
		assert (cudaSuccess == cudaMemset(d_delta, 0, sizeof(int) * n_count));
		set_int_edge<<<1,1>>>(d_dist, --h_dist);
		while(h_dist > 1) {
			backward_edge<<<grid, threads>>>(d_v, d_e, d_d, d_sigma, d_delta, d_dist, e_count);
			cudaThreadSynchronize();
			set_int_edge<<<1,1>>>(d_dist, --h_dist);
			CudaCheckError();
		}
		backsum_edge<<<grid2, threads2>>>(i, d_d,  d_delta, d_bc, n_count);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}
	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_v);
	cudaFree(d_e);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	return 0;
}

