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

#ifndef __BUCKET__H
#define __BUCKET__H


#ifdef __cplusplus
/* if C++, define the rest of this header file as extern C */
extern "C" {
#endif


    /* This is ID-less bucket data structure to save memory
       and hence speed up bucket updates. It assumes IDs are
       0-based indices without any gap */
typedef struct S
{
  struct S* prev;
  struct S* next;
} Bucket_element;


typedef unsigned int eType;
typedef int vType;

typedef struct arg
{
    Bucket_element **buckets; /* actual pointers to bucket heads */
    Bucket_element *elements; /* for direct access to bucket elements
                                 elements[id] is the id-th element */
    vType          nb_elements;
    vType            max_value;
    vType              *values; /* needed for update, incase bucket head
                                 changed. */

  vType current_min_value;
} Bucket;

/* value == INT_MAX means not present in bucket */
void Zoltan_Bucket_Insert(Bucket* bs, vType id, vType value);

void Zoltan_Bucket_Update(Bucket* bs, vType id, vType new_value);

#define Zoltan_Bucket_DecVal(bs, id) Zoltan_Bucket_Update(bs, id, (bs)->values[id]-1)

/*returns -1 if empty*/
vType Zoltan_Bucket_PopMin(Bucket* bs);

Bucket Zoltan_Bucket_Initialize(vType max_value, vType nb_element);

void Zoltan_Bucket_Free(Bucket* bs);

#ifdef __cplusplus
} /* closing bracket for extern "C" */
#endif
    


#endif
