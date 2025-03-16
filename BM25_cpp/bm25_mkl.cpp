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
    
    float* bm25_scores = new float[num_docs];  // Allocate output array

    #pragma omp parallel for schedule(static)
    for (int d = 0; d < num_docs; d++) {
        float local_score = 0.0f;
        float* tile_scores = new float[TILE_SIZE];  // Dynamically allocate tile buffer

        for (int t_start = 0; t_start < num_terms; t_start += TILE_SIZE) {
            int t_end = min(t_start + TILE_SIZE, num_terms);

            for (int t = t_start; t < t_end; t++) {
                float tf = term_frequencies[d * num_terms + t];
                float idf = idf_scores[t];
                float doc_len_norm = (1.0 - b) + (b * (doc_lengths[d] / avg_doc_length));
                tile_scores[t - t_start] = idf * ((tf * (k1 + 1)) / (tf + (k1 * doc_len_norm)));
            }

            // Create a vector of ones to sum tile_scores using cblas_sdot
            float ones[TILE_SIZE];
            for (int i = 0; i < (t_end - t_start); i++) {
                ones[i] = 1.0f;
            }

            // Sum the tile scores using cblas_sdot
            local_score += cblas_sdot(t_end - t_start, tile_scores, 1, ones, 1);
        }

        bm25_scores[d] = local_score;  // Store the final BM25 score for document d
        delete[] tile_scores;          // Free the tile buffer
    }

    return bm25_scores;  // Caller must free this memory
}



// Simple test program to validate BM25 scoring
int main() {
    const int num_docs = 3;
    const int num_terms = 5;  // Vocabulary size

    // Term Frequencies (TF) for 3 documents, 5 terms each
    float term_frequencies[num_docs * num_terms] = {
        3, 0, 1, 2, 0,  // Document 1
        0, 2, 0, 1, 3,  // Document 2
        1, 1, 1, 1, 1   // Document 3
    };

    // Precomputed IDF scores for each term
    float idf_scores[num_terms] = {1.2, 0.8, 1.5, 1.0, 1.3};

    // Document lengths (for simplicity, all documents have length 100)
    float doc_lengths[num_docs] = {100.0, 100.0, 100.0};
    float avg_doc_length = 100.0;  // With all doc lengths 100

    // Compute BM25 scores
    float* bm25_results = compute_bm25(term_frequencies, idf_scores, doc_lengths, avg_doc_length, num_docs, num_terms);

    // Print BM25 Scores along with expected values (for reference)
    cout << "\n🔹 BM25 Scores for Test Documents:\n";
    cout << "Document 1 Score: " << bm25_results[0] << "   /* expected ~4.93 */\n";
    cout << "Document 2 Score: " << bm25_results[1] << "   /* expected ~4.31 */\n";
    cout << "Document 3 Score: " << bm25_results[2] << "   /* expected ~5.80 */\n";

    // Free dynamically allocated memory
    delete[] bm25_results;

    cout << "\n✅ BM25 Test Completed. Press Enter to exit..." << endl;
    cin.get();
    
    return 0;
}

