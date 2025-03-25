## Project Overview

This project introduces a custom Tokenizer that extends the Hugging Face tokenizer framework, 
integrating advanced retrieval capabilities with inbuilt Retrieval-Augmented Generation (RAG) support. 
It combines FAISS-based vector search with a custom BM25 ranking algorithm accelerated using Intel's Math Kernel Library (MKL), 
allowing for high-performance hybrid retrieval. The tokenizer is designed to work seamlessly within modern NLP pipelines, 
offering both dense and sparse retrieval modes, making it ideal for knowledge-enhanced tasks like document search, 
context injection for LLMs, and domain-specific chatbots.

The project was tested using CPU-based inferencing with the meta-llama/Llama-3.2-1B-Instruct model. 
The pipeline operates as follows: user input is first tokenized using the custom tokenizer, 
during which FAISS and the MKL-accelerated BM25 retrieval are executed directly at the tokenization level. 
The resulting contextually enriched input is then passed to the model for generation. 
This design enables fast, on-the-fly retrieval without requiring external pre-processing, 
making the system lightweight and easily deployable even in CPU-constrained environments (This specific model Was deployed with CPU inferencing).


# Advantages

Tight Integration: Retrieval (FAISS and BM25) happens within the tokenization step, streamlining the workflow and reducing latency by eliminating separate preprocessing pipelines.

Hybrid Retrieval: Supports both dense (FAISS) and sparse (BM25) retrieval, allowing flexible use cases and improving recall across diverse query types.

CPU-Friendly Deployment: Designed with CPU-based inferencing in mind, making it ideal for low-resource or edge environments.

Modular and Extensible: Built as a subclass of the Hugging Face tokenizer, it fits naturally into transformer workflows and can be easily extended or integrated with existing models.

MKL Acceleration: The use of Intel MKL significantly boosts BM25 performance, making sparse retrieval practical even on large corpora.

# Disadvantages

Coupled Architecture: By embedding retrieval logic in the tokenizer, the design becomes less modular, making it harder to swap out components independently (e.g., using a different retriever).

Tokenization Overhead: Retrieval adds computation time during tokenization, which could slow down batch processing or streaming applications.

Complex Debugging: Errors in the retrieval layer may appear during tokenization, making it harder to isolate and troubleshoot issues.

Limited GPU Optimization: While FAISS supports GPU acceleration, the current design focuses on CPU usage — potentially leaving out performance gains for GPU-rich setups.

Model-Agnostic Challenges: Passing retrieved content directly into the model assumes a prompt structure that may not be optimal for all LLM architectures.


## File Structure



📁 File Structure
+ README.md — Project documentation (you're editing this!)

+ requirements.txt — Frozen Python dependencies

+ retrieval_data.tar.gz — Compressed snapshot of retrieval dataset

+ wikipedia_category_articles.csv — Source dataset used for indexing (from dataset_aquisition.  ipynb)

📂 BM25_cpp — C++ source code for BM25 with MKL acceleration
+ Function_Profiling/ — Performance profiling scripts (tested using DrMemory VSCode Profiler)

+ bm25_bindings.cpp — Pybind11 wrapper code

+ bm25_mkl.cpp — Core MKL-based BM25 implementation

+ bm25_mkl.h — C++ header for BM25 functions

📂 build — Compiled artifacts and built libraries (Important)
+ bm25_bindings.* — Build output (.exp, .lib, .obj)

+ bm25_mkl.* — Build output (.obj, .pyd for Python module)

+ build_test.ipynb — Test notebook for verifying compiled module

+ *.dll — Required runtime DLLs (e.g., Intel MKL, Python)

+ req_paths.py — Helper to set DLL & module paths (Important)

📂 retrieval_data — Serialized precomputed data (decompressed from retrieval_data.tar.gz)
+ article_map.json — Maps article IDs to text or categories

+ bm25_*.npy / *.json — Token frequencies, lengths, IDF values for BM25

+ faiss_*.npy / .faiss — Dense embeddings and FAISS index

+ tokenized_corpus.json — Tokenized version of the full corpus

📂 retrieval_lib — Python library code
+ __init__.py — Library init file

+ retrieval.py — Retrieval logic (BM25, FAISS integration)

+ tokenizer.py — Custom tokenizer subclassing HuggingFace


🧪 Jupyter Notebooks
+ data_preprocess.ipynb — Preprocessing pipeline (creating retrieval data)

+ dataset_aquisition.ipynb — Script for downloading/curating data (getting test data)

+ llama.ipynb — Inference notebook using Llama 3.2 1B CPU (main)


## (ignore this ore open readme file to view non sauished structure)

RAG_TOKENIZER/
│
├── README.md                         # Project documentation (you're editing this!)
├── requirements.txt                  # Frozen Python dependencies
├── retrieval_data.tar.gz             # Compressed snapshot of retrieval dataset
├── wikipedia_category_articles.csv   # Source dataset used for indexing (from dataset_aquisition.ipynb)
│
├── BM25_cpp/                         # C++ source code for BM25 with MKL acceleration
│   ├── Function_Profiling/           # Performance profiling scripts (tested using DrMemory VSCode Profiler)
│   ├── bm25_bindings.cpp             # Pybind11 wrapper code
│   ├── bm25_mkl.cpp                  # Core MKL-based BM25 implementation
│   └── bm25_mkl.h                    # C++ header for BM25 functions
│
├── build/                            # Compiled artifacts and built libraries (*Important*)
│   ├── bm25_bindings.*               # Build output (.exp, .lib, .obj)
│   ├── bm25_mkl.*                    # Build output (.obj, .pyd for Python module)
│   ├── build_test.ipynb              # Test notebook for verifying compiled module
│   ├── *.dll                         # Required runtime DLLs (e.g., Intel MKL, Python)
│   └── req_paths.py                  # Helper to set DLL & module paths (*Important*)
│
├── retrieval_data/                   # Serialized precomputed data (decompressed from retrieval_data.tar.gz)
│   ├── article_map.json              # Maps article IDs to text or categories
│   ├── bm25_*.npy / *.json           # Token frequencies, lengths, IDF values for BM25
│   ├── faiss_*.npy / .faiss          # Dense embeddings and FAISS index
│   └── tokenized_corpus.json         # Tokenized version of the full corpus
│
├── retrieval_lib/                    # Python library code
│   ├── __init__.py                   # Library init file
│   ├── retrieval.py                  # Retrieval logic (BM25, FAISS integration)
│   └── tokenizer.py                  # Custom tokenizer subclassing HuggingFace
│
├── trash/                            # Deprecated or unused code (temporary)
│
├── data_preprocess.ipynb             # Preprocessing pipeline (creating retrieval data)
├── dataset_aquisition.ipynb          # Script for downloading/curating data (Getting Test Data)
└── llama.ipynb                       # Inference notebook using Llama 3.2 1B CPU (main)
