import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoModelForCausalLM

if __name__ == '__main__':
    EMBEDDING_MODEL_NAME = "naver/splade-v3"
    COMP_MODEL_NAME = "naver/cocom-v1-128-mistral-7b"

    # initialize model embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Use cosine similarity normalization.
    )

    # load docs
    print("Loading combined documents...")
    combined_output_path = "/group/jmearlesgrp/data/KILT/processed_docs/combined_processed_docs.pkl"
    with open(combined_output_path, "rb") as f:
        combined_docs = pickle.load(f)

    # set fraction for testing
    use_fraction = 1.0  # Change this value as needed.
    num_docs = int(len(combined_docs) * use_fraction)
    combined_docs = combined_docs[:num_docs]
    print(f"Using {num_docs} documents for vector database creation.")

    # save FAISS db
    vector_db_dir = Path("/group/jmearlesgrp/data/KILT/vector_dbs")
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    
    comp_model = AutoModelForCausalLM.from_pretrained(COMP_MODEL_NAME, trust_remote_code=True).to("cuda")

    batch_size = 100000  # Adjust the batch size as needed.
    batch_files = []      # Keep track of saved FAISS batch files.

    # process vectors
    for i in tqdm(range(0, len(combined_docs), batch_size), desc="Processing FAISS batches"):
        batch_docs = combined_docs[i : i + batch_size]
        faiss_index = FAISS.from_documents(batch_docs, embedding_model, distance_strategy=DistanceStrategy.COSINE)

        batch_file = vector_db_dir / f"faiss_batch_{i}.faiss"
        faiss_index.save_local(batch_file)
        batch_files.append(batch_file)

    print(f"Saved {len(batch_files)} FAISS vector databases.")

    print("Merging FAISS vector databases...")
    merged_faiss = None

    for batch_file in tqdm(batch_files, desc="Loading FAISS batches"):
        current_faiss = FAISS.load_local(batch_file, comp_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)

        if merged_faiss is None:
            merged_faiss = current_faiss
        else:
            merged_faiss.merge_from(current_faiss)

    # Save the final merged FAISS index.
    final_faiss_path = Path("/group/jmearlesgrp/data/KILT/faiss_index_large")
    merged_faiss.save_local(final_faiss_path)

    print(f"Final FAISS index saved to {final_faiss_path}.")
    
    # delete individual batch files to free disk space.
    for batch_file in batch_files:
        try:
            batch_file.unlink()
            print(f"Deleted {batch_file}")
        except Exception as e:
            print(f"Could not delete {batch_file}: {e}")

    try:
        vector_db_dir.rmdir()
        print(f"Deleted directory {vector_db_dir}")
    except Exception as e:
        print(f"Directory {vector_db_dir} not removed: {e}")
