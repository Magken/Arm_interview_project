#ifndef BM25_MKL_H
#define BM25_MKL_H

#include <iostream>
#include <cmath>
#include <mkl.h>   // Intel MKL Header
#include <omp.h>   // OpenMP for parallelism

// BM25 function declaration (used in both C++ and Python bindings)
float* compute_bm25(const float* term_frequencies, 
                     const float* idf_scores, 
                     const float* doc_lengths, 
                     float avg_doc_length, 
                     int num_docs, 
                     int num_terms);

#endif  // BM25_MKL_H
