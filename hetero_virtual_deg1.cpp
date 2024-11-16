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
#include <omp.h>
#include <sys/time.h>
#include <stdio.h>

using namespace std;

void lastOperations (int ncount, float* h_bc);
void lastOperations_coalesced (int ncount, float* h_bc);
void allocate (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int* h_weight);
void allocate_coalesced (int* h_vmap, int* h_vptrs, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride, int e_count, int virn_count, int* h_weight);
void one_source(int source, int n_count, int virn_count);
void one_source_coalesced(int source, int n_count, int virn_count);
void bc_cpu_deg1_one_source (int* xadj, int* adj, int nVtx, int nz,  float* bc, int* weight, int source, int* bfsorder, int* Pred, int* endpred, int* level, int* sigma, float* delta );

int bc_virtual_deg1_hetero (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc, int* h_weight,
		int* xadj, int* adj, int nz, int* weight) {

	allocate(h_vmap, h_vptrs, h_vjs, n_count, e_count, virn_count, h_weight);


	float ** bctemps;

#pragma omp parallel
	{
		bool usegpu = true;
		if (getenv("NOGPU")){
			usegpu = false;
		}
		int tid = omp_get_thread_num();

#pragma omp master
		{
			bctemps = new float*[omp_get_num_threads()];
			bctemps[0] = h_bc;
			if (usegpu)
				std::cout<<"using gpu and "<<omp_get_num_threads()-1<<" CPU thread"<<std::endl;
			else
				std::cout<<"using"<<omp_get_num_threads()<<" CPU thread"<<std::endl;
		}

		float* temp_bc;
		temp_bc = new float[n_count];
		float* bc;
		bc = new float [n_count];
		for (int i=0; i< n_count; ++i)
			bc[i] = 0;

		int* bfsorder;
		int* Pred;
		int* endpred;
		int* level;
		int* sigma;
		float* delta;

#pragma omp barrier

		if (tid != 0 || !usegpu)
		{
			bfsorder = new int[n_count];
			Pred = new int[xadj[n_count]];
			endpred = new int[n_count];
			level = new int[n_count];
			sigma = new int[n_count];
			delta = new float[n_count];

			bctemps[tid] = bc;
		}



#pragma omp for schedule (dynamic,1)
		for(int i = 0; i < min (nb, n_count); i++){
			if (tid == 0 && usegpu) {
				one_source(i, n_count, virn_count);
			}
			else
			{
				bc_cpu_deg1_one_source (xadj, adj, n_count, nz, bctemps[tid], weight,
						i,
						bfsorder, Pred, endpred, level, sigma, delta );
			}
		}
#pragma omp barrier

		if (tid != 0 || ! usegpu)
		{
			delete[] bfsorder;
			delete[] Pred;
			delete[] level;
			delete[] sigma;
			delete[] delta;
			delete[] endpred;
		}
		else
			lastOperations (n_count, temp_bc);

		if (tid == 0) {
		  for (int i = 0; i < n_count; ++i) {
		    h_bc[i] += temp_bc[i];
		  }
		}
		
#pragma omp barrier

#pragma omp for schedule (dynamic,128)
		for (int i = 0; i < n_count; ++i)
		  {

		    for (int t = 1; t < omp_get_num_threads(); ++t) {
		      h_bc[i] += bctemps[t][i];
		    }
		  }


		if (tid != 0 || !usegpu)
		{
			delete[] bc;
		}
		else
			delete[] bctemps;
	}

	return 0;
}

int bc_virtual_coalesced_deg1_hetero (int* h_vmap, int* h_vptrs, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride,
		int e_count, int virn_count, int nb, float *h_bc, int* h_weight, int* xadj, int* adj, int nz, int* weight) {

	allocate_coalesced(h_vmap, h_vptrs, h_xadj, h_vjs, n_count, h_startoffset, h_stride, e_count, virn_count, h_weight);

	float ** bctemps;

#pragma omp parallel
	{
		bool usegpu = true;
		if (getenv("NOGPU")){
			usegpu = false;
		}
		int tid = omp_get_thread_num();

#pragma omp master
		{
			bctemps = new float*[omp_get_num_threads()];
			bctemps[0] = h_bc;
			if (usegpu)
				std::cout<<"using gpu and "<<omp_get_num_threads()-1<<" CPU thread"<<std::endl;
			else
				std::cout<<"using"<<omp_get_num_threads()<<" CPU thread"<<std::endl;
		}

		float* temp_bc;
		temp_bc = new float[n_count];
		float* bc;
		bc = new float [n_count];
		for (int i=0; i< n_count; ++i)
			bc[i] = 0;
		int* bfsorder;
		int* Pred;
		int* endpred;
		int* level;
		int* sigma;
		float* delta;

#pragma omp barrier

		if (tid != 0 || !usegpu)
		{
			bfsorder = new int[n_count];
			Pred = new int[xadj[n_count]];
			endpred = new int[n_count];
			level = new int[n_count];
			sigma = new int[n_count];
			delta = new float[n_count];

			bctemps[tid] = bc;
		}


#pragma omp for schedule (dynamic,1)
		for(int i = 0; i < min (nb, n_count); i++){
			if (tid == 0 && usegpu) {
				one_source_coalesced(i, n_count, virn_count);
			}
			else
			{
				bc_cpu_deg1_one_source (xadj, adj, n_count, nz, bctemps[tid], weight,
						i,
						bfsorder, Pred, endpred, level, sigma, delta );
			}
		}

#pragma omp barrier

		if (tid != 0 || ! usegpu)
		{
			delete[] bfsorder;
			delete[] Pred;
			delete[] level;
			delete[] sigma;
			delete[] delta;
			delete[] endpred;
		}
		else
			lastOperations (n_count, temp_bc);

		if (tid == 0)
			for (int i = 0; i < n_count; ++i) {
				h_bc[i] += temp_bc[i];
			}
#pragma omp barrier

#pragma omp for schedule (dynamic,128)
		for (int i = 0; i < n_count; ++i)
		{
			for (int t = 1; t < omp_get_num_threads(); ++t)
				h_bc[i] += bctemps[t][i];
		}


		if (tid != 0 || !usegpu)
		{
			delete[] bc;
		}
		else
			delete[] bctemps;
	}

	return 0;
}
