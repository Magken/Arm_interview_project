import os
import sys
import numpy as np
import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

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
    """
    Custom tokenizer with BM25 & FAISS retrieval (MKL-optimized).
    This class wraps a real Hugging Face tokenizer and delegates any missing attributes.
    """

    def __init__(self, model_name, max_bm25_results=5, max_faiss_results=5, 
                 use_faiss_gpu=False, num_faiss_threads=None):
        """
        Args:
          - model_name (str): Hugging Face model name.
          - max_bm25_results (int): Number of top BM25 results to retrieve.
          - max_faiss_results (int): Number of top FAISS results to retrieve.
          - use_faiss_gpu (bool): If True, moves FAISS index to GPU (default: False).
          - num_faiss_threads (int or None): Number of CPU threads for FAISS. If None, uses all available cores.
        """
        # Load the underlying tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)  # Used for FAISS embeddings

        # Get available CPU cores and set number of threads
        max_cpu_cores = os.cpu_count()
        self.num_faiss_threads = min(num_faiss_threads or max_cpu_cores, max_cpu_cores)

        # Set retrieval hyperparameters
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

        # FAISS Setup (CPU with MKL or Optional GPU)
        print(f"✅ Initializing FAISS with {self.num_faiss_threads} CPU threads (Max cores: {max_cpu_cores})")
        faiss.omp_set_num_threads(self.num_faiss_threads)
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        if self.use_faiss_gpu and torch.cuda.is_available():
            print("✅ Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Load article mapping
        with open(ARTICLE_MAP_PATH, "r") as f:
            self.article_map = json.load(f)

        # Set common tokenizer properties so they are available to users
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token = self.tokenizer.bos_token
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_llama_embedding(self, text):
        """Generate a fixed-size embedding using the underlying model (LLaMA)."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
        pooled_embedding = hidden_states.mean(dim=1).numpy().astype("float32")
        return pooled_embedding.reshape(1, -1)

    def retrieve_bm25(self, query):
        """Retrieve top BM25 results for a given query."""
        if self.max_bm25_results == 0:
            return []
        query_tokens = self.tokenizer.tokenize(query)
        query_indices = [self.vocab[token] for token in query_tokens if token in self.vocab]
        if not query_indices:
            return []
        tf_query = self.tf_array[:, query_indices]
        idf_query = self.idf_scores[query_indices]
        tf_flat = tf_query.flatten()
        bm25_scores = bm25_mkl.compute_bm25(
            tf_flat,
            idf_query,
            self.token_lengths,
            self.avg_token_length,
            self.tf_array.shape[0],
            len(query_indices)
        )
        sorted_indices = np.argsort(bm25_scores)[::-1]
        top_bm25_articles = [self.article_map[str(idx)] for idx in sorted_indices[:self.max_bm25_results]]
        return top_bm25_articles

    def add_retrieval_context(self, history):
        """
        Modify the last user input in the history with BM25 (and optionally FAISS) retrieved context.
        Expects history as a list of dictionaries with keys "role" and "content".
        """
        if not history or history[-1]["role"] != "user":
            return history  # No modification needed if there's no user input

        user_input = history[-1]["content"]
        # Retrieve BM25 context (FAISS is disabled if max_faiss_results == 0)
        bm25_results = self.retrieve_bm25(user_input)
        faiss_results = []  # You can add FAISS results similarly if needed
        combined_results = list(set(bm25_results + faiss_results))
        retrieval_context = " ".join(combined_results)
        modified_input = f"Text:( {retrieval_context}). According to the text {user_input} ?"
        history[-1]["content"] = modified_input

        print(history)

        return history

    def tokenize(self, input_data, *args, **kwargs):
        """
        If input_data is a list (chat history), modify its last user message using retrieval augmentation,
        then join the messages into a single string to tokenize.
        If input_data is a string, use it directly.
        """
        if isinstance(input_data, list):
            modified_history = self.add_retrieval_context(input_data)
            text_to_tokenize = " ".join([msg["content"] for msg in modified_history])
        elif isinstance(input_data, str):
            text_to_tokenize = input_data
        else:
            raise TypeError("Input must be a list (chat history) or a string.")
        return self.tokenizer(text_to_tokenize, *args, **kwargs)

    def decode(self, tokenized_input, *args, **kwargs):
        """
        Decode tokenized output back to text.
        
        Supports:
        - tokenized_input as a dict with "input_ids"
        - tokenized_input as a list of token IDs
        - tokenized_input as a torch.Tensor (converted to a list)
        - tokenized_input as an object with an 'input_ids' attribute (e.g. BatchEncoding)
        """
        # If tokenized_input has an attribute "input_ids", use it.
        if hasattr(tokenized_input, "input_ids"):
            input_ids = tokenized_input.input_ids
            return self.tokenizer.decode(input_ids, *args, **kwargs)
        elif isinstance(tokenized_input, dict) and "input_ids" in tokenized_input:
            return self.tokenizer.decode(tokenized_input["input_ids"], *args, **kwargs)
        elif isinstance(tokenized_input, list):
            return self.tokenizer.decode(tokenized_input, *args, **kwargs)
        elif isinstance(tokenized_input, torch.Tensor):
            return self.tokenizer.decode(tokenized_input.tolist(), *args, **kwargs)
        else:
            raise TypeError(f"Unexpected input type for decode: {type(tokenized_input)}. "
                            "Expected dict, list, torch.Tensor, or an object with an 'input_ids' attribute.")


    def __getattr__(self, name):
        """
        Delegate attribute lookup to the underlying tokenizer if not found.
        This makes methods like convert_ids_to_tokens available.
        """
        return getattr(self.tokenizer, name)
