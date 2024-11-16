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

#ifdef __cplusplus
/* if C++, define the rest of this header file as extern C */
extern "C" {
#endif

#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include "bucket.h"


void Zoltan_Bucket_Insert(Bucket* bs, vType id, vType value)
{
#if 0
    assert (bs != NULL);
    assert (value >= 0);
    assert (value <= bs->max_value);    
    assert (id >= 0);
    assert (id < bs->nb_elements);
#endif
    
    bs->values[id] = value;
    
    bs->elements[id].prev = NULL;
    bs->elements[id].next = bs->buckets[value];
    
    if (bs->buckets[value] != NULL) 
        bs->buckets[value]->prev = &(bs->elements[id]);
    else if (bs->current_min_value > value)
        bs->current_min_value = value;

    bs->buckets[value] = &(bs->elements[id]);

}

void Zoltan_Bucket_Update(Bucket* bs, vType id, vType new_value)
{
    vType old_value = bs->values[id];

    if (old_value == INT_MAX)
        return;
    
#if 0  
    assert (bs != NULL);
    assert (new_value >= 0);
    assert (new_value <= bs->max_value);        
    assert (id >= 0);
    assert (id < bs->nb_elements);
#endif
  
    bs->values[id] = new_value;


    if (bs->elements[id].prev == NULL)
        bs->buckets[old_value] = bs->elements[id].next;
    else
        bs->elements[id].prev->next = bs->elements[id].next;
  
    if (bs->elements[id].next != NULL)
        bs->elements[id].next->prev = bs->elements[id].prev;

    Zoltan_Bucket_Insert(bs, id, new_value);
}

vType Zoltan_Bucket_PopMin(Bucket* bs)
{
    vType id;

#if 0  
    assert (bs != NULL);
    assert (bs->current_min_value >= 0);
#endif

    for (; bs->current_min_value<=bs->max_value; bs->current_min_value++) {
        if (bs->buckets[bs->current_min_value] != NULL) {
            id = bs->buckets[bs->current_min_value] - bs->elements;
            bs->buckets[bs->current_min_value] = bs->buckets[bs->current_min_value]->next;
            if (bs->buckets[bs->current_min_value] != NULL) {
                bs->buckets[bs->current_min_value]->prev = NULL;
            }
            return id;
        }
    }
    return -1;
}

Bucket Zoltan_Bucket_Initialize(vType max_value, vType nb_element)
{
    Bucket bs;
    vType i;

#if 0  
    assert (max_value>=0);
    assert (nb_element>=0);
#endif

    bs.buckets = (Bucket_element **) malloc(sizeof(Bucket_element *) * (max_value+1));
    bs.elements = (Bucket_element *) malloc(sizeof(Bucket_element) * nb_element);
    bs.values = (vType *) malloc(sizeof(vType) * nb_element);
    bs.max_value = max_value;
    bs.nb_elements = nb_element;

    if (bs.buckets == NULL || bs.elements == NULL || bs.values == NULL) {
        free(bs.values);
        free(bs.buckets);
        free(bs.elements);
    } else {
        for (i=0; i<=max_value; i++)
            bs.buckets[i] = NULL;

        for (i=0; i<nb_element; i++) {
            bs.elements[i].prev = NULL;
            bs.elements[i].next = NULL;
        }
    }
    bs.current_min_value = max_value+1;
    return bs;
}

void Zoltan_Bucket_Free(Bucket* bs)
{
    free(bs->values);
    free(bs->buckets);
    free(bs->elements);
}


#ifdef __cplusplus
} /* closing bracket for extern "C" */
#endif
