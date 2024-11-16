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
 For license info, please see the README.txt and LICENSE.txt files in
 the main directory.
---------------------------------------------------------------------
*/

============================================================
Betwenness Centrality on GPUs and Heterogenous Architectures
============================================================

- Source files and their brief descriptions are as follows:
  LICENSE.txt:              contains LICENSE information
  README.txt:               this file
  bc_vertex.cu:             vertex-based GPU parallelization functions 
  bc_edge.cu:               edge-based GPU parallelization functions
  bc_virtual.cu:            virtual-vertex based GPU parallelization, with and without strided access, functions
  bucket.c:                 utility functions for degree-1 vertex removal
  bucket.h:                 header for bucket.c
  common.h:                 GPU-related macros
  compile:                  script to compile source code
  cuda_common.h:            GPU error-checking function
  gpu-bc.cpp:               main function and CPU functions for BC
  graph.h:                  graph read functions
  hetero_virtual_deg1.cpp:  heteregenous computing functions for virtual-vertex parallelization, with and without strided(coalesced) access
  preproc.cu:               degree-1 vertex removal and ordering functions
  timestamp.hpp:            timing functions
  ulib.c:                   some utility functions
  ulib.h:                   header for ulib.c

- To compile the code, make sure you have a version of nvcc and type
    $> ./compile

- To run, give the following arguments:
    <input filename> 
    <ordering enable>                  // 1 to enable,
                                       // 0 to disable ordering
    <degree1 vertex enable>            // 1 to enable,
                                       // 0 to disable degree-1 vertex removal
    <mode>                             // 0 for CPU,
                                       // 1 for vertex-based GPU parallelism,
                                       // 2 for edge-based GPU parallelism
                                       // 3 for virtual-vertex based GPU paralellism
                                       // 4 for virtual-vertex based GPU with strided access
    <number of bfs's>                  // number of bfs runs to be executed.
                                       // Largest component of the graph is used for bc computation.
    <maximum degree of virtual vertex> // used for virtual-vertex based variants. 
                                       // Details can be found in the paper.

- For <input filename>, Chaco, Matrix Market and SNAP formats are supported. Chaco
  format input files should have an extension of ".graph", Matrix Market
  files should have an extension of ".mtx" and SNAP formats should have an
  extension of ".txt".

- Degree-1 vertex removal is not available for edge-based parallelism variant.

- Betweenness centrality values of vertices are written to "bc_out.txt" file.

- Heterogenous computing is available when degree-1 vertex removal is enabled
  for virtual-vertex based parallelism variants, i.e., mode 3 and 4.
  OMP_NUM_THREADS variable should be set in command line.

- Example run where ordering and degree-1 vertex removal techniques
  are enabled when virtual-vertex based GPU parallelism with strided
  access is used (1000 number of bfs and max degree of virtual vertex is 4):
     $> ./gpu_bc bc-seq-brandes power.graph 1 1 4 1000 4

- If you use gpuBC, please cite:
 	"A. E. Sariyuce, K. Kaya, E. Saule, U. V. Catalyurek. 
 	 "Betweenness Centrality on GPUs and Heterogeneous Architectures", 
 	 Workshop on General Purpose Processing Using GPUs (GPGPU), in conjunction with ASPLOS, 2013.


- For any question or problem, please contact:
    aerdem@bmi.osu.edu
    kamer@bmi.osu.edu 
    esaule@bmi.osu.edu
    umit@bmi.osu.edu
