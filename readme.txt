Main Program (main.py)	                      Calls retrieval functions from the library
Custom Library (retrieval.py)	              Implements FAISS & BM25 classes, calls the C++ BM25 function
C++ BM25 (bm25_mkl.cpp)	                      Implements fast BM25 scoring using Intel MKL
Python-C++ Bridge (bm25_wrapper.cpp)	      Uses Pybind11 to expose C++ BM25 to Python