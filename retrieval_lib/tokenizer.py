import os
import numpy as np
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import sys
from .retrieval import Functionalities  # Import retrieval functionality

# Ensure Windows can locate required DLLs (adjust paths as needed)
os.add_dll_directory("E:\\Intel\\oneAPI\\mkl\\latest\\bin")
os.add_dll_directory("E:\\Coding Stuff\\Arm_interview_project\\build")
os.add_dll_directory("C:\\Users\\Magjun\\AppData\\Local\\Programs\\Python\\Python311")
sys.path.append("E:\\Coding Stuff\\Arm_interview_project\\build")

import bm25_mkl

# Ensure retrieval data is extracted
Functionalities.extract_retrieval_data()

# Load BM25 & FAISS Data
RETRIEVAL_DIR = "retrieval_data"
BM25_TF_PATH = os.path.join(RETRIEVAL_DIR, "bm25_tf.npy")
BM25_IDF_PATH = os.path.join(RETRIEVAL_DIR, "bm25_idf.npy")
BM25_TOKEN_LENGTHS_PATH = os.path.join(RETRIEVAL_DIR, "bm25_token_lengths.npy")
BM25_AVG_TOKEN_LENGTH_PATH = os.path.join(RETRIEVAL_DIR, "bm25_avg_token_length.json")
VOCAB_PATH = os.path.join(RETRIEVAL_DIR, "vocab.json")
ARTICLE_MAP_PATH = os.path.join(RETRIEVAL_DIR, "article_map.json")
FAISS_INDEX_PATH = os.path.join(RETRIEVAL_DIR, "faiss_token_index.faiss")


class CustomRetrieverTokenizer:
    """Custom tokenizer with BM25 & FAISS retrieval (MKL-optimized)."""

    def __init__(self, model_name, max_bm25_results=5, max_faiss_results=5, 
                 use_faiss_gpu=False, num_faiss_threads=None):
        """
        Args:
        - model_name (str): Hugging Face model name.
        - max_bm25_results (int): Number of top BM25 results to retrieve.
        - max_faiss_results (int): Number of top FAISS results to retrieve.
        - use_faiss_gpu (bool): If True, moves FAISS index to GPU (default: False).
        - num_faiss_threads (int or None): Number of CPU threads for FAISS. 
          If None, uses all available cores but limits to system max.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)  # Load LLaMA for FAISS embeddings

        # Get total available CPU cores
        max_cpu_cores = os.cpu_count()
        self.num_faiss_threads = min(num_faiss_threads or max_cpu_cores, max_cpu_cores)  # Prevent exceeding CPU cores

        # Set hyperparameters
        self.max_bm25_results = max_bm25_results
        self.max_faiss_results = max_faiss_results
        self.use_faiss_gpu = use_faiss_gpu

        # Load BM25 Data
        self.tf_array = np.load(BM25_TF_PATH)
        self.idf_scores = np.load(BM25_IDF_PATH)
        self.token_lengths = np.load(BM25_TOKEN_LENGTHS_PATH)
        with open(BM25_AVG_TOKEN_LENGTH_PATH, "r") as f:
            self.avg_token_length = json.load(f)["avg_token_length"]
        with open(VOCAB_PATH, "r") as f:
            self.vocab = json.load(f)

        # **FAISS Setup (CPU with MKL or Optional GPU)**
        print(f"✅ Initializing FAISS with {self.num_faiss_threads} CPU threads (Max cores: {max_cpu_cores})")
        faiss.omp_set_num_threads(self.num_faiss_threads)  # Set FAISS CPU threads
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)  # Load FAISS index

        # Optionally Move FAISS to GPU
        if self.use_faiss_gpu and torch.cuda.is_available():
            print("✅ Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Load article mapping
        with open(ARTICLE_MAP_PATH, "r") as f:
            self.article_map = json.load(f)

    def get_llama_embedding(self, text):
        """Generate a fixed-size embedding using LLaMA."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # Extract hidden states
        pooled_embedding = hidden_states.mean(dim=1).numpy().astype('float32')
        return pooled_embedding.reshape(1, -1)  # Ensure FAISS shape (1, embedding_dim)

    def retrieve_bm25(self, query):
        """Retrieve top BM25 results given a query."""
        if self.max_bm25_results == 0:
            return []  # Skip BM25 retrieval if set to 0

        query_tokens = self.tokenizer.tokenize(query)

        # Get indices for query tokens (skip tokens not in vocab)
        query_indices = [self.vocab[token] for token in query_tokens if token in self.vocab]

        if not query_indices:
            return []

        # Select only query-relevant TF and IDF scores
        tf_query = self.tf_array[:, query_indices]
        idf_query = self.idf_scores[query_indices]

        # Flatten TF for BM25 function
        tf_flat = tf_query.flatten()

        # Compute BM25 Scores
        bm25_scores = bm25_mkl.compute_bm25(
            tf_flat,
            idf_query,
            self.token_lengths,
            self.avg_token_length,
            self.tf_array.shape[0],  # Number of documents
            len(query_indices)  # Number of query terms
        )

        # Sort Results by Score
        sorted_indices = np.argsort(bm25_scores)[::-1]  # Highest score first
        top_bm25_articles = [self.article_map[str(idx)] for idx in sorted_indices[:self.max_bm25_results]]

        return top_bm25_articles

    def retrieve_faiss(self, query):
        """Retrieve top FAISS results given a query."""
        if self.max_faiss_results == 0:
            return []  # Skip FAISS retrieval if set to 0

        query_embedding = self.get_llama_embedding(query)

        # Ensure dimensions match
        assert query_embedding.shape[1] == self.faiss_index.d, "Dimension mismatch! Fix embedding shape."

        # Perform FAISS search
        distances, indices = self.faiss_index.search(query_embedding, self.max_faiss_results)

        # Retrieve articles using indices
        retrieved_articles = [self.article_map[str(idx)] for idx in indices[0]]

        return retrieved_articles

    def add_retrieval_context(self, query):
        """Combine BM25 & FAISS retrieved information with query."""
        bm25_results = self.retrieve_bm25(query)
        faiss_results = self.retrieve_faiss(query)

        # Merge results and remove duplicates
        combined_results = list(set(bm25_results + faiss_results))

        # Append to query
        retrieval_context = " ".join(combined_results)
        enriched_query = f"Found relevant info: {retrieval_context}. Use it to answer user query: {query}"

        return enriched_query

    def tokenize(self, text, *args, **kwargs):
        """Override tokenization to include retrieval augmentation."""
        enriched_text = self.add_retrieval_context(text)
        return self.tokenizer(enriched_text, *args, **kwargs)

    def decode(self, tokenized_input):
        """Decode tokenized output back to text."""
        return self.tokenizer.decode(tokenized_input["input_ids"])