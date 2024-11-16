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

#include <assert.h>
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
#include "cuda_common.h"

using namespace std;

int *d_vmap, *d_vptrs, *d_vjs, *d_d, *d_sigma, *d_dist, h_dist;
float *d_delta, *d_bc;
int* d_weight;
bool *d_continue;
dim3 grid, threads, grid2, threads2;

//for deg1 coalesced
int *d_xadj;
int *d_startoffset;
int *d_stride;

__global__ void forward_virtual (int* d_vmap, int* d_vptrs, int* d_vjs, int *d_d, int *d_sigma, bool *d_continue, int *d_dist, int virn_count) {
	int vu = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		int u = d_vmap[vu];
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u] == *d_dist) {
			int end = d_vptrs[vu + 1];
			for(int p = d_vptrs[vu]; p < end; p++) {
				int w = d_vjs[p];
				if(d_d[w] == -1) {
					d_d[w] = *d_dist + 1;
					*d_continue = 1;
				}
				if(d_d[w] == *d_dist + 1) {
					atomicAdd(&d_sigma[w], d_sigma[u]);
				}
			}
		}
	}
}

__global__ void forward_virtual_coalesced (int* d_vmap, int* d_vptrs, int* d_vjs, int *d_d, int *d_sigma, bool *d_continue, int *d_dist, int virn_count, int *d_stride, int *d_startoffset, int *d_xadj) {
	int vu = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		int u = d_vmap[vu];
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u] == *d_dist) {
			int end = d_xadj[u + 1];
			int stride = d_stride[u];
			for(int p = d_startoffset[vu]; p < end; p+=stride) {
				int w = d_vjs[p];
				if(d_d[w] == -1) {
					d_d[w] = *d_dist + 1;
					*d_continue = 1;
				}
				if(d_d[w] == *d_dist + 1) {
					atomicAdd(&d_sigma[w], d_sigma[u]);
				}
			}
		}
	}
}

__global__ void backward_virtual_coalesced (int* d_vmap, int* d_vptrs, int* d_vjs, int *d_d, float *d_delta, int *d_dist, int virn_count, int *d_stride, int *d_startoffset, int *d_xadj){
	int vu = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		int u = d_vmap[vu];
		if(d_d[u] == *d_dist - 1) {
			int end = d_xadj[u + 1];
			int stride = d_stride[u];
			float sum = 0;
			for(int p = d_startoffset[vu]; p < end; p+=stride) {
				int w = d_vjs[p];
				if(d_d[w] == *d_dist ) {
					sum += d_delta[w];
				}
			}
			atomicAdd(&d_delta[u], sum);
		}
	}
}

__global__ void backward_virtual (int* d_vmap, int* d_vptrs, int* d_vjs, int *d_d, float *d_delta, int *d_dist, int virn_count){
	int vu = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		int u = d_vmap[vu];
		if(d_d[u] == *d_dist - 1) {
			int end = d_vptrs[vu+1];
			float sum = 0;
			for(int p = d_vptrs[vu]; p < end; p++) {
				int w = d_vjs[p];
				if(d_d[w] == *d_dist ) {
					sum += d_delta[w];
				}
			}
			atomicAdd(&d_delta[u], sum);
		}
	}
}

__global__ void intermediate_virtual (int *d_d, int *d_sigma, float *d_delta, int *d_dist, int n_count) {
	int u = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(u < n_count) {
		d_delta[u] = 1.0f / d_sigma[u];
	}
}

__global__ void intermediate_virtual_deg1 (int *d_d, int *d_sigma, float *d_delta, int *d_dist, int n_count, int* d_weight) {
	int u = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(u < n_count) {
		d_delta[u] = d_weight[u] / (float)d_sigma[u];
	}
}

__global__ void backsum_virtual_deg1 (int s, int *d_d, float *d_delta, int *d_sigma, float *d_bc, int n_count, int* d_weight){
	int tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += (d_delta[tid] * d_sigma[tid] - 1) * d_weight[s];
	}
}

__global__ void backsum_virtual (int s, int *d_d, float *d_delta, int *d_sigma, float *d_bc, int n_count){
	int tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid] * d_sigma[tid] - 1;
	}
}

__global__ void init_virtual (int s, int *d_d, int *d_sigma, int n_count, int* d_dist){
	int i = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
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

__global__ void set_int (int* dest, int val){
	*dest = val;
}

int bc_virtual (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc) {
	int *d_vmap, *d_vptrs, *d_vjs, *d_d, *d_sigma, *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(int) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(int) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(int) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(int) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(int) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(int) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(int)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(int)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	int threads_per_block = virn_count;
	int blocks = 1;
	if(virn_count > MTS){
		blocks = (int)ceil(virn_count/(double)MTS);
		blocks = (int)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);

	int threads_per_block2 = n_count;
	int blocks2 = 1;
	if(n_count > MTS){
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
		init_virtual<<<grid2,threads2>>>(i, d_d, d_sigma, n_count, d_dist);
		cudaThreadSynchronize();

		CudaCheckError();
#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif


		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
			forward_virtual<<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count);
			cudaThreadSynchronize();
			CudaCheckError();
			set_int<<<1,1>>>(d_dist, ++h_dist);
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

#ifdef TIMER
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif
		}while(h_continue);
#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		set_int<<<1,1>>>(d_dist, --h_dist);
		intermediate_virtual<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count);
		cudaThreadSynchronize();
		while(h_dist > 1) {
			backward_virtual<<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count);
			cudaThreadSynchronize();
			set_int<<<1,1>>>(d_dist, --h_dist);
			CudaCheckError();
		}
		backsum_virtual<<<grid2, threads2>>>(i, d_d,  d_delta, d_sigma, d_bc, n_count);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	return 0;
}

int bc_virtual_coalesced (int* h_vmap, int* h_vptrs, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride, int e_count, int virn_count, int nb, float *h_bc) {
	int *d_vmap, *d_vptrs, *d_vjs, *d_d, *d_sigma, *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	int *d_xadj;
	int *d_startoffset;
	int *d_stride;

	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(int) *  virn_count));
	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(int) * virn_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(int) * e_count));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(int) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_xadj, sizeof(int) * (n_count + 1)));
	assert (cudaSuccess == cudaMemcpy(d_xadj, h_xadj, sizeof(int) * (n_count + 1), cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_startoffset, sizeof(int) * (virn_count)));
	assert (cudaSuccess == cudaMemcpy(d_startoffset, h_startoffset, sizeof(int) * (virn_count), cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_stride, sizeof(int) * (n_count)));
	assert (cudaSuccess == cudaMemcpy(d_stride, h_stride, sizeof(int) * (n_count), cudaMemcpyHostToDevice));


	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(int)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(int)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	int threads_per_block = virn_count;
	int blocks = 1;
	if(virn_count > MTS){
		blocks = (int)ceil(virn_count/(double)MTS);
		blocks = (int)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);

	int threads_per_block2 = n_count;
	int blocks2 = 1;
	if(n_count > MTS){
		blocks2 = (int)ceil(n_count/(double)MTS);
		blocks2 = (int)ceil(sqrt((float)blocks2));
		threads_per_block2 = MTS;
	}
	dim3 grid2;
	grid2.x = blocks2;
	grid2.y = blocks2;
	dim3 threads2(threads_per_block2);

	cout<<"cuda parameters: "<<blocks<<" "<<threads_per_block<<" "<<blocks2<<" "<<threads_per_block2<<endl;

	cout<<"coalesced"<<std::endl;

#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif
	for(int i = 0; i < min(nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_virtual<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif
		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
			forward_virtual_coalesced<<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count, d_stride, d_startoffset, d_xadj);
			cudaThreadSynchronize();
			set_int<<<1,1>>>(d_dist, ++h_dist);
			CudaCheckError();
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

#ifdef TIMER
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif
		}while(h_continue);
#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		set_int<<<1,1>>>(d_dist, --h_dist);
		intermediate_virtual<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count);
		cudaThreadSynchronize();
		while(h_dist > 1) {
			backward_virtual_coalesced<<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count, d_stride, d_startoffset, d_xadj);
			cudaThreadSynchronize();
			CudaCheckError();
			set_int<<<1,1>>>(d_dist, --h_dist);
		}
		backsum_virtual<<<grid2, threads2>>>(i, d_d,  d_delta, d_sigma, d_bc, n_count);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	CudaCheckError();

	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	return 0;
}

int bc_virtual_deg1 (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc, int* h_weight) {
	int *d_vmap, *d_vptrs, *d_vjs, *d_d, *d_sigma, *d_dist, h_dist;
	float *d_delta, *d_bc;
	int* d_weight;
	bool h_continue, *d_continue;

	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(int) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(int) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(int) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(int) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(int) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(int) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(int)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_weight, sizeof(int) * n_count));
	assert (cudaSuccess == cudaMemcpy(d_weight, h_weight, sizeof(int) * n_count, cudaMemcpyHostToDevice)); // weight array
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(int)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float) * n_count));
	assert (cudaSuccess == cudaMemcpy(d_bc, h_bc, sizeof(int) * n_count, cudaMemcpyHostToDevice)); // bc array

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	int threads_per_block = virn_count;
	int blocks = 1;
	if(virn_count > MTS){
		blocks = (int)ceil(virn_count/(double)(MTS));
		blocks = (int)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);

	int threads_per_block2 = n_count;
	int blocks2 = 1;
	if(n_count > MTS){
		blocks2 = (int)ceil(n_count/(double)(MTS));
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
	for(int i = 0; i < min (nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_virtual<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif
		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			cudaMemset(d_continue, 0, sizeof(bool));
			forward_virtual<<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count);
			cudaThreadSynchronize();
			set_int<<<1,1>>>(d_dist, ++h_dist);
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

		set_int<<<1,1>>>(d_dist, --h_dist);
		intermediate_virtual_deg1<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count, d_weight);
		cudaThreadSynchronize();
		while(h_dist > 1) {
			backward_virtual<<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count);
			cudaThreadSynchronize();
			set_int<<<1,1>>>(d_dist, --h_dist);
			CudaCheckError();
		}

		backsum_virtual_deg1<<<grid2, threads2>>>(i, d_d,  d_delta, d_sigma, d_bc, n_count, d_weight);
		cudaThreadSynchronize();

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	return 0;
}

void lastOperations (int n_count, float* h_bc) {
	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
}

int createVirtualCSR(int* ptrs, int* js, int nov, int* vmap, int* virptrs, int maxload, int permuteAdj) {
	int vcount = 0, deg, nvirtual, remaining, dif;
	int* temp = (int*)malloc(sizeof(int) * ptrs[nov]);
	int* temp2 = (int*)malloc(sizeof(int) * ptrs[nov]);

	cout<<"virtualizing vertices! "<< nov<<endl;

	virptrs[0] = 0;
	for(int i = 0; i < nov; i++) {
		deg = ptrs[i+1] - ptrs[i];
		//		cout<<deg<<endl;
		nvirtual = deg / maxload;
		remaining = deg % maxload;

		for(int j = 0; j < nvirtual; j++) { /* these are for full virtual vertices */
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + maxload;
			vcount++;
		}

		if(remaining > 0) {
			if(nvirtual > 0) {
				dif = (maxload - remaining) / 2;
				virptrs[vcount] -= dif;
				remaining += dif;
			}
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + remaining;
			vcount++;
		}
	}

	if(permuteAdj) { /* scatters the nonzeros to virtual vertices */
		int p, start, to;
		memcpy(temp2, virptrs, sizeof(int) * (vcount+1));
		start = 0;
		for(int i = 0; i < nov; i++) {
			while(vmap[start] < i && start < vcount) {
				start++;
			}
			if(start < vcount && vmap[start] == i) {
				to = start;
				for(p = ptrs[i]; p < ptrs[i+1];) {
					temp[temp2[to]++] = js[p++];

					while(p < ptrs[i+1]) {
						to++;
						if(to == vcount || vmap[to] != i) to = start;
						if(temp2[to] < virptrs[to+1]) break;
					}
				}
			}
		}
	}

	printf("number of virtual vertices %d\n", vcount);

	if(ptrs[nov] != virptrs[vcount]) {
		printf("VIRTUAL: %d != %d\n", ptrs[nov], virptrs[vcount]);
		exit(1);
	}

	free(temp);
	free(temp2);
	return vcount;
}

int createVirtualCoalescedCSR(int* ptrs, int* js, int nov, int* vmap, int* virptrs, int* startoffset, int* stride, int maxload, int permuteAdj) {
	int vcount = 0, deg, nvirtual, remaining, dif;
	int* temp = (int*)malloc(sizeof(int) * ptrs[nov]);
	int* temp2 = (int*)malloc(sizeof(int) * ptrs[nov]);

	cout<<"virtualizing vertices! "<< nov<<endl;

	virptrs[0] = 0;
	for(int i = 0; i < nov; i++) {
		deg = ptrs[i+1] - ptrs[i];
		//		cout<<deg<<endl;
		nvirtual = deg / maxload;
		remaining = deg % maxload;

		stride[i] = nvirtual+(remaining>0); //total number of virtual vertex for vertex i

		for(int j = 0; j < nvirtual; j++) { /* these are for full virtual vertices */
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + maxload;


			startoffset[vcount] = ptrs[i]+j;

			vcount++;
		}

		if(remaining > 0) {
			startoffset[vcount] = ptrs[i]+nvirtual;
			if(nvirtual > 0) {
				dif = (maxload - remaining) / 2;
				virptrs[vcount] -= dif;
				remaining += dif;
			}
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + remaining;
			vcount++;
		}
	}

	if(permuteAdj) { /* scatters the nonzeros to virtual vertices */
		int p, start, to;
		memcpy(temp2, virptrs, sizeof(int) * (vcount+1));
		start = 0;
		for(int i = 0; i < nov; i++) {
			while(vmap[start] < i && start < vcount) {
				start++;
			}
			if(start < vcount && vmap[start] == i) {
				to = start;
				for(p = ptrs[i]; p < ptrs[i+1];) {
					temp[temp2[to]++] = js[p++];

					while(p < ptrs[i+1]) {
						to++;
						if(to == vcount || vmap[to] != i) to = start;
						if(temp2[to] < virptrs[to+1]) break;
					}
				}
			}
		}
	}

#ifdef DEBUG
	printf("number of virtual vertices %d\n", vcount);
#endif

	if(ptrs[nov] != virptrs[vcount]) {
		printf("VIRTUAL: %d != %d\n", ptrs[nov], virptrs[vcount]);
		exit(1);
	}

	free(temp);
	free(temp2);
	return vcount;
}

void allocate (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int* h_weight) {

	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(int) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(int) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(int) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(int) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(int) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(int) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(int)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_weight, sizeof(int) * n_count));
	assert (cudaSuccess == cudaMemcpy(d_weight, h_weight, sizeof(int) * n_count, cudaMemcpyHostToDevice)); // weight array
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(int)));


	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float) * n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float) * n_count)); // bc array

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	int threads_per_block = virn_count;
	int blocks = 1;
	if(virn_count > MTS) {
		blocks = (int)ceil(virn_count/(double)(MTS));
		blocks = (int)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	grid.x = blocks;
	grid.y = blocks;
	threads = threads_per_block;

	int threads_per_block2 = n_count;
	int blocks2 = 1;
	if(n_count > MTS){
		blocks2 = (int)ceil(n_count/(double)(MTS));
		blocks2 = (int)ceil(sqrt((float)blocks2));
		threads_per_block2 = MTS;
	}
	grid2.x = blocks2;
	grid2.y = blocks2;
	threads2 = threads_per_block2;

	cout<<"cuda parameters: "<<blocks<<" "<<threads_per_block<<" "<<blocks2<<" "<<threads_per_block2<<endl;


}

void one_source(int source, int n_count, int virn_count) {

	int h_dist = 0;
	bool h_continue;
	init_virtual<<<grid,threads>>>(source, d_d, d_sigma, n_count, d_dist);

	do{

		assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
		forward_virtual<<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count);
		cudaThreadSynchronize();
		CudaCheckError();
		set_int<<<1,1>>>(d_dist, ++h_dist);
		cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);

	} while(h_continue);

	set_int<<<1,1>>>(d_dist, --h_dist);
	intermediate_virtual_deg1<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count, d_weight);
	cudaThreadSynchronize();
	while(h_dist > 1) {
		backward_virtual<<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count);
		cudaThreadSynchronize();
		CudaCheckError();
		set_int<<<1,1>>>(d_dist, --h_dist);
	}

	backsum_virtual_deg1<<<grid2, threads2>>>(source, d_d,  d_delta, d_sigma, d_bc, n_count, d_weight);
	cudaThreadSynchronize();
}

void allocate_coalesced (int* h_vmap, int* h_vptrs, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride, int e_count, int virn_count, int* h_weight) {
	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(int) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(int) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(int) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(int) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(int) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(int) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(int)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_weight, sizeof(int) * n_count));
	assert (cudaSuccess == cudaMemcpy(d_weight, h_weight, sizeof(int) * n_count, cudaMemcpyHostToDevice)); // weight array
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(int)));


	assert (cudaSuccess == cudaMalloc((void **)&d_xadj, sizeof(int) * (n_count + 1)));
	assert (cudaSuccess == cudaMemcpy(d_xadj, h_xadj, sizeof(int) * (n_count + 1), cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_startoffset, sizeof(int) * (virn_count)));
	assert (cudaSuccess == cudaMemcpy(d_startoffset, h_startoffset, sizeof(int) * (virn_count), cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_stride, sizeof(int) * (n_count)));
	assert (cudaSuccess == cudaMemcpy(d_stride, h_stride, sizeof(int) * (n_count), cudaMemcpyHostToDevice));


	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float) * n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(int) * n_count)); // bc array

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	int threads_per_block = virn_count;
	int blocks = 1;
	if(virn_count > MTS) {
		blocks = (int)ceil(virn_count/(double)(MTS));
		blocks = (int)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	grid.x = blocks;
	grid.y = blocks;
	threads = threads_per_block;

	int threads_per_block2 = n_count;
	int blocks2 = 1;
	if(n_count > MTS){
		blocks2 = (int)ceil(n_count/(double)(MTS));
		blocks2 = (int)ceil(sqrt((float)blocks2));
		threads_per_block2 = MTS;
	}
	grid2.x = blocks2;
	grid2.y = blocks2;
	threads2 = threads_per_block2;

	cout<<"cuda parameters: "<<blocks<<" "<<threads_per_block<<" "<<blocks2<<" "<<threads_per_block2<<endl;


}

void one_source_coalesced(int source, int n_count, int virn_count) {

	int h_dist = 0;
	bool h_continue;
	init_virtual<<<grid,threads>>>(source, d_d, d_sigma, n_count, d_dist);

	do{

		assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
		forward_virtual_coalesced<<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count, d_stride, d_startoffset, d_xadj);
		cudaThreadSynchronize();
		CudaCheckError();
		set_int<<<1,1>>>(d_dist, ++h_dist);
		cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);

	} while(h_continue);

	set_int<<<1,1>>>(d_dist, --h_dist);
	intermediate_virtual_deg1<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count, d_weight);
	cudaThreadSynchronize();
	while(h_dist > 1) {
		backward_virtual_coalesced<<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count, d_stride, d_startoffset, d_xadj);
		cudaThreadSynchronize();
		CudaCheckError();
		set_int<<<1,1>>>(d_dist, --h_dist);
	}

	backsum_virtual_deg1<<<grid2, threads2>>>(source, d_d,  d_delta, d_sigma, d_bc, n_count, d_weight);
	cudaThreadSynchronize();
}

void lastOperations_coalesced (int n_count, float* h_bc) {
	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
}
