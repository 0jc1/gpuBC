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

#ifndef __COMMON_H__
#define __COMMON_H__

#define MTS 256
#define WARP 16
#define MAXLOAD 4

#define MAXLINE 128*(1024*1024)

//#define TIMER
//#define DEBUG

#include <stdio.h>


#endif
