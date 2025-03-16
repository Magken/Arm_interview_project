#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bm25_mkl.h"

namespace py = pybind11;

py::array_t<float> compute_bm25_py(py::array_t<float> term_frequencies, 
                                   py::array_t<float> idf_scores, 
                                   py::array_t<float> doc_lengths, 
                                   float avg_doc_length, 
                                   int num_docs, 
                                   int num_terms) {
    
    auto tf_buf = term_frequencies.request();
    auto idf_buf = idf_scores.request();
    auto doc_len_buf = doc_lengths.request();

    float* tf_ptr = static_cast<float*>(tf_buf.ptr);
    float* idf_ptr = static_cast<float*>(idf_buf.ptr);
    float* doc_len_ptr = static_cast<float*>(doc_len_buf.ptr);

    // Call the C++ function
    float* result = compute_bm25(tf_ptr, idf_ptr, doc_len_ptr, avg_doc_length, num_docs, num_terms);

    // Convert result to Python array
    py::array_t<float> output({num_docs});
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);

    for (int i = 0; i < num_docs; i++) {
        out_ptr[i] = result[i];
    }

    delete[] result;  // ✅ Free C++ allocated memory

    return output;
}

PYBIND11_MODULE(bm25_mkl, m) {
    m.doc() = "BM25 MKL Optimized Module"; 
    m.def("compute_bm25", &compute_bm25_py, "Compute BM25 scores using MKL");
}
