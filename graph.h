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

#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>
#include <map>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <list>
#include <omp.h>
#include <sys/time.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include "ulib.h"
#include "timestamp.hpp"

#define MAXLINE 128*(1024*1024)

using namespace std;

typedef struct pair {
	long long f;
	long long s;
} Pair;

int pcmp(const void *v1, const void *v2){
	long long diff = (((Pair *)v1)->f - ((Pair *)v2)->f);
	if (diff != 0)
		return diff;
	else
		return (((Pair *)v1)->s - ((Pair *)v2)->s);
}

template <typename VtxType, typename WeightType>
void ReadGraphFromFile(FILE *fpin, VtxType *numofvertex, VtxType **pxadj, VtxType **padjncy, WeightType **padjncyw, WeightType **ppvw) {
	VtxType *xadj, *adjncy,  nvtxs, nedges, fmt, readew, readvw, edge, i, k, ncon;
	WeightType*adjncyw=NULL, *pvw=NULL;
	char *line;

	line = (char *)malloc(sizeof(char)*(MAXLINE+1));

	do {
		fgets(line, MAXLINE, fpin);
	} while (line[0] == '%' && !feof(fpin));

	if (feof(fpin)){
		printf("empty graph!!!");
		exit(1);
	}

	fmt = 0;
	{
		std::string s = line;
		std::stringstream ss (s);
		ss>>nvtxs>>nedges>>fmt>>ncon;
	}
	*numofvertex = nvtxs;

	readew = (fmt%10 > 0);
	readvw = ((fmt/10)%10 > 0);
	if (fmt >= 100) {
		printf("invalid format");
		exit(1);
	}

	nedges *=2;

	xadj = *pxadj = imalloc(nvtxs+2, "ReadGraph: xadj");
	adjncy = *padjncy = imalloc(nedges, "ReadGraph: adjncy");
	if (padjncyw)
		adjncyw = *padjncyw = imalloc(nedges, "ReadGraph: adjncyw");
	if (ppvw)
		pvw = *ppvw = imalloc(nvtxs+1, "ReadGraph: adjncyw");

	for (xadj[0]=0, k=0, i=0; i<nvtxs; i++) {
		char *oldstr=line, *newstr=NULL;
		int  ewgt=1, vw=1;

		do {
			fgets(line, MAXLINE, fpin);
		} while (line[0] == '%' && !feof(fpin));

		if (strlen(line) >= MAXLINE-5){
			printf("\nBuffer for fgets not big enough!\n");
			exit(1);
		}

		if (readvw) {
			vw = (int)strtol(oldstr, &newstr, 10);
			oldstr = newstr;
		}

		if (ppvw)
			pvw[i] = vw;

		for (;;) {
			edge = (int)strtol(oldstr, &newstr, 10) -1;
			oldstr = newstr;

			if (readew) {
				ewgt = (int)strtol(oldstr, &newstr, 10);
				oldstr = newstr;
			}

			if (edge < 0) {
				break;
			}

			if (edge==i){
				printf("Self loop in the graph for vertex %d\n", i);
				exit(1);
			}
			adjncy[k] = edge;
			if (padjncyw)
				adjncyw[k] = ewgt;
			k++;
		}
		xadj[i+1] = k;
	}

	if (k != nedges){
		printf("k(%d)!=nedges(%d) and i:%d", k, nedges,i);
		exit(1);
	}

	free(line);

	return;
}

template <typename VtxType, typename WeightType>
void ReadGraphFromMMFile(FILE *matfp, VtxType *numofvertex, VtxType **pxadj, VtxType **padjncy, WeightType **padjncyw, WeightType **ppvw, bool do_mapping, long** reverse_map) {

	Pair *coords, *new_coords;
	int m, n, itemp, jtemp;
	long long nnz, tnnz, i, j, onnz;;
	int *xadj, *adj, *adjncyw, *pvw;

	int maxLineLength = 1000000;
	int value = 0;
	char line[maxLineLength];
	int num_items_read;

	/* set return null parameter values, in case we exit with errors */
	m = nnz = 0;

	/* now continue scanning until you reach the end-of-comments */
	do {
		if (fgets(line, 1000000, matfp) == NULL)
			value = 1;
	} while (line[0] == '%');

	/* line[] is either blank or has M,N, nz */
	if (sscanf(line, "%d %d %lld", &m, &n, &nnz) == 3) {
		value = 1;
	}
	else {
		do {
			num_items_read = fscanf(matfp, "%d %d %lld", &m, &n, &nnz);
			if (num_items_read == EOF)
				return;
		}
		while (num_items_read != 3);
		value = 0;
	}

	*reverse_map = (long*) malloc (sizeof(long) * n);
	//	printf("matrix banner is read %d - %d, %lld nnz\n", m, n, nnz);
	coords = (Pair*) malloc(sizeof(Pair) * 2 * nnz);

	tnnz = 0;
	for(i = 0; i < nnz; i++) {
		fscanf(matfp, "%d %d\n", &itemp, &jtemp);

		if(itemp != jtemp) {
			coords[tnnz].f = itemp - 1;
			coords[tnnz++].s = jtemp - 1;
			coords[tnnz].f = jtemp - 1;
			coords[tnnz++].s = itemp - 1;
		}
	}
	//	printf("matrix is read %d - %d, %lld nnz with duplicates\n", m, n, tnnz);

	qsort(coords, tnnz, sizeof(Pair), pcmp);

	onnz = 1;
	for(i = 1; i < tnnz; i++) {
		if(coords[i].f != coords[onnz-1].f || coords[i].s != coords[onnz-1].s) {
			coords[onnz].f = coords[i].f;
			coords[onnz++].s = coords[i].s;
		}
	}

	*numofvertex = n;
	xadj = *pxadj = (int*) malloc((n+1) * sizeof(int));
	adj = *padjncy = (int*) malloc(onnz * sizeof(int));
	if (padjncyw)
		adjncyw = *padjncyw = imalloc (nnz, "ReadGraph: adjncyw");
	if (ppvw)
		pvw = *ppvw = imalloc (n+1, "ReadGraph: adjncyw");

	if (do_mapping) {
		map<long, int> reed;
		map<long, int>::iterator reed_it;
		new_coords = (Pair*) malloc(sizeof(Pair) * 2 * nnz);
		long vno = 0;
		// map the ids
		for(i = 0; i < onnz; i++) {
			long temp = coords[i].f;
			reed_it = reed.find(temp);
			if (reed_it == reed.end()) {
				reed.insert (make_pair (temp, vno));
				(*reverse_map)[vno] = temp;
				new_coords[i].f = vno++;
			}
			else
				new_coords[i].f = reed_it->second;

			temp = coords[i].s;
			reed_it = reed.find(temp);
			if (reed_it == reed.end()) {
				reed.insert (make_pair (temp, vno));
				(*reverse_map)[vno] = temp;
				new_coords[i].s = vno++;
			}
			else
				new_coords[i].s = reed_it->second;
		}
	}

	vector<vector<int> > entire_graph;
	entire_graph.resize(n);
	for(i = 0; i < onnz; i++) {
		if (do_mapping)
			entire_graph[new_coords[i].f].push_back(new_coords[i].s);
		else
			entire_graph[coords[i].f].push_back(coords[i].s);
	}
	xadj[0] = 0;
	j = 0;
	for(i = 1; i < n+1; i++) {
		xadj[i] = xadj[i-1] + entire_graph[i-1].size();
		for (unsigned int k = 0; k < entire_graph[i-1].size(); k++) {
			adj[j++] = entire_graph[i-1][k];
		}
	}

	//	printf("matrix is read %d - %d, %lld onnz\n", m, n, onnz);
	free(coords);
	for(i = 0; i < m; i++)
		entire_graph[i].clear();

	entire_graph.clear();

	return;
}

static int really_read(std::istream& is, char* buf, size_t global_size) {
	char* temp2 = buf;
	while (global_size != 0) {
		is.read(temp2, global_size);
		size_t s = is.gcount();
		if (!is)
			return -1;

		global_size -= s;
		temp2 += s;
	}
	return 0;
}

template <typename VtxType, typename EdgeType, typename WeightType>
void ReadBinary(char *filename, VtxType *numofvertex_r, VtxType *numofvertex_c, EdgeType **pxadj, VtxType **padj, WeightType **padjw, WeightType **ppvw) {

	if (ppvw != NULL) {
		cerr<<"vertex weight is unsupported"<<std::endl;
		return;
	}

	std::ifstream in (filename);

	if (!in.is_open()) {
		cerr<<"can not open file:"<<filename<<std::endl;
		return;
	}

	int vtxsize; //in bytes
	int edgesize; //in bytes
	int weightsize; //in bytes

	//reading header
	in.read((char *)&vtxsize, sizeof(int));
	in.read((char *)&edgesize, sizeof(int));
	in.read((char *)&weightsize, sizeof(int));


	printf("vtxsize: %d\n", vtxsize);
	printf("edgesize: %d\n", edgesize);
	printf("weightsize: %d\n", weightsize);


	cout<<"vtxsize: "<<vtxsize<<endl;
	cout<<"edgesize: "<<edgesize<<endl;
	cout<<"weightsize: "<<weightsize<<endl;

	cout<<"erdem"<<endl;

	if (!in) {
		cerr<<"IOError"<<std::endl;
		return;
	}

	if (vtxsize != sizeof(VtxType)) {
		cerr<<"Incompatible VertexSize."<<endl;
		return;
	}

	if (edgesize != sizeof(EdgeType)) {
		cerr<<"Incompatible EdgeSize."<<endl;
		return;
	}

	if (weightsize != sizeof(WeightType)) {
		cerr<<"Incompatible WeightType."<<endl;
		return;
	}

	//reading should be fine from now on.
	in.read((char*)numofvertex_r, sizeof(VtxType));
	in.read((char*)numofvertex_c, sizeof(VtxType));
	EdgeType nnz;
	in.read((char*)&nnz, sizeof(EdgeType));
	if (numofvertex_c <=0 || numofvertex_r <=0 || nnz <= 0) {
		cerr<<"graph makes no sense"<<endl;
		return;
	}

	cout<<"nVtx: "<<*numofvertex_r<<endl;
	cout<<"nVtx: "<<*numofvertex_c<<endl;
	cout<<"nEdge: "<<nnz<<endl;
	printf("nVtx: %ld, nVtx: %ld, nEdge: %ld\n", *numofvertex_r, *numofvertex_c, nnz);

	*pxadj = (EdgeType*) malloc (sizeof(EdgeType) * (*numofvertex_r+1));
	*padj =  (VtxType*) malloc (sizeof(VtxType) * (nnz));


	if (padjw) {
		*padjw = new WeightType[nnz];
	}

	int err = really_read(in, (char*)*pxadj, sizeof(EdgeType)*(*numofvertex_r+1));
	err += really_read(in, (char*)*padj, sizeof(VtxType)*(nnz));
	if (padjw)
		err += really_read(in, (char*)*padjw, sizeof(WeightType)*(nnz));
	if (!in || err != 0) {
		cerr<<"IOError"<<endl;
	}

	return;
}

template <typename VtxType, typename WeightType>
bool ReadGraph(char *filename, VtxType *numofvertex, VtxType **pxadj, VtxType **padjncy, int** ptadj, WeightType **padjncyw, WeightType **ppvw, long** reverse_map) {
	FILE *fpin = ufopen(filename, "r", "main: fpin");

	char * pch = NULL;
	pch = strstr (filename,".");
	char t1[10];
	char t2[10];
	char t3[10];
	strcpy (t1, ".graph");
	strcpy (t2, ".mtx");
	strcpy (t3, ".bin");

	if (pch == NULL)
		ReadGraphFromMMFile (fpin, numofvertex, pxadj, padjncy, padjncyw, ppvw, true, reverse_map);
	else if (strcmp (pch, t1) == 0)
		ReadGraphFromFile (fpin, numofvertex, pxadj, padjncy, padjncyw, ppvw);
	else if (strcmp (pch, t2) == 0)
		ReadGraphFromMMFile (fpin, numofvertex, pxadj, padjncy, padjncyw, ppvw, false, reverse_map);
	else if (strcmp (pch, t3) == 0)
		ReadBinary<int, int, int> (filename, numofvertex, numofvertex, pxadj, padjncy, NULL, NULL);
	else
		ReadGraphFromMMFile (fpin, numofvertex, pxadj, padjncy, padjncyw, ppvw, true, reverse_map);

	int* adj = *padjncy;
	int* xadj = *pxadj;
	int n = *numofvertex;
	*ptadj = (int*) malloc(sizeof(int) * xadj[n]);
	int* tadj = *ptadj;

	int i;
	int* degs = (int*)malloc(sizeof(int) * n);
	int* myedges = (int*)malloc(sizeof(int) * xadj[n]);

	memcpy(degs, xadj, sizeof(VtxType) * n);
	int ptr, j;
	for(i = 0; i < n; i++) {
		for(ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
			j = adj[ptr];
			myedges[degs[j]++] = i;
		}
	}

	for(i = 0; i < n; i++) {
		if(xadj[i+1] != degs[i]) {
			printf("something is wrong, %d %d %d\n", i, xadj[i+1], degs[i]);
			exit(1);
		}
	}

	memcpy(adj, myedges, sizeof(VtxType) * xadj[n]);
	for(i = 0; i < n; i++) {
		for(ptr = xadj[i]+1; ptr < xadj[i+1]; ptr++) {
			if(adj[ptr] <= adj[ptr-1]) {
				printf("is not sorted\n");
				exit(1);
			}
		}
	}
	memcpy(degs, xadj, sizeof(VtxType) * n);
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

	ufclose(fpin);

	if (pch == NULL)
		return true;
	else
		return false;
}

#endif
