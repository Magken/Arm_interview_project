#include "bm25_mkl.h"
#include <iostream>
#include <cmath>
#include <mkl.h>   // Intel MKL Header
#include <omp.h>   // OpenMP for parallelism

using namespace std;

// BM25 Parameters
const float k1 = 1.5;
const float b = 0.75;
const int TILE_SIZE = 16;  // Block size for tiling

// Compute BM25 scores using Intel MKL with Blocking (Tiling) and OpenMP
float* compute_bm25(const float* term_frequencies, 
                     const float* idf_scores, 
                     const float* doc_lengths, 
                     float avg_doc_length, 
                     int num_docs, 
                     int num_terms) {
    
    float* bm25_scores = new float[num_docs];  // ✅ Allocate dynamically

    #pragma omp parallel for schedule(dynamic)
    for (int d = 0; d < num_docs; d++) {
        float local_score = 0.0f;

        float tile_scores[TILE_SIZE];  // ✅ Use stack allocation

        for (int t_start = 0; t_start < num_terms; t_start += TILE_SIZE) {
            int t_end = min(t_start + TILE_SIZE, num_terms);

            for (int t = t_start; t < t_end; t++) {
                float tf = term_frequencies[d * num_terms + t];
                float idf = idf_scores[t];
                float doc_len_norm = (1.0 - b) + (b * (doc_lengths[d] / avg_doc_length));

                tile_scores[t - t_start] = idf * ((tf * (k1 + 1)) / (tf + (k1 * doc_len_norm)));
            }

            local_score += cblas_sdot(t_end - t_start, tile_scores, 1, idf_scores + t_start, 1);
        }

        bm25_scores[d] = local_score;  // ✅ Store the final BM25 score
    }

    return bm25_scores;  // ✅ Caller must delete[] this memory
}



// Example Usage
int main() {
    const int num_docs = 2;
    const int num_terms = 64;  // Increased terms for better tiling

    // ✅ Allocate term frequencies dynamically
    float* term_frequencies = new float[num_docs * num_terms];
    
    // Generate random term frequencies
    for (int i = 0; i < num_docs * num_terms; i++) {
        term_frequencies[i] = (rand() % 5) + 1;  // Random values between 1-5
    }

    // ✅ Allocate idf scores dynamically
    float* idf_scores = new float[num_terms];
    for (int i = 0; i < num_terms; i++) {
        idf_scores[i] = 1.0f + (rand() % 5) * 0.1f;  // Random IDF scores
    }

    // ✅ Allocate document lengths dynamically
    float* doc_lengths = new float[num_docs];
    doc_lengths[0] = 100.0;
    doc_lengths[1] = 150.0;
    float avg_doc_length = 125.0;

    // Compute BM25 scores
    float* bm25_results = compute_bm25(term_frequencies, idf_scores, doc_lengths, avg_doc_length, num_docs, num_terms);

    // Print the results
    for (int i = 0; i < num_docs; i++) {
        cout << "BM25 Score for Document " << i << ": " << bm25_results[i] << endl;
    }

    // ✅ Free all dynamically allocated memory
    delete[] term_frequencies;
    delete[] idf_scores;
    delete[] doc_lengths;
    delete[] bm25_results;

    // Pause to view output
    cout << "Computation complete. Press Enter to exit..." << endl;
    cin.get();

    return 0;
}