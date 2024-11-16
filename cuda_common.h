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

#ifndef CUDACOMMON
#define CUDACOMMON

inline void CudaCheckError() {
  // ask CUDA for the last error to occur (if one exists)
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // something's gone wrong
    // print out the CUDA error as a string
    printf("CUDA Error: %s\n", cudaGetErrorString(error));

    // we can't recover from the error -- exit the program
    //return 1;
  }
}

#endif
