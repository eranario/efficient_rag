import os
import pickle
import datasets
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument

global_tokenizer = None
cache_dir = "/group/jmearlesgrp/data/KILT/cache"
RAW_COLLECTION = datasets.load_dataset("dmrau/kilt-128", cache_dir=cache_dir)

def init_worker(tokenizer_name):
    global global_tokenizer
    global_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def create_tokens(doc, tokenizer, chunk_size, chunk_overlap=0):

    if isinstance(doc, dict):
        text = doc.get('content', '')
        metadata = doc
    else:
        text = doc
        metadata = {}
    tokens = tokenizer.encode(text, add_special_tokens=False)
    splits = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        token_chunk = tokens[start:end]
        chunk_text = tokenizer.decode(token_chunk, skip_special_tokens=True)
        splits.append(LangchainDocument(page_content=chunk_text, metadata=metadata))
        start += chunk_size - chunk_overlap
    return splits

def process_doc(doc):

    return create_tokens(doc, global_tokenizer, chunk_size=128)

def save_batch(batch_docs, batch_index, output_dir):
    output_path = os.path.join(output_dir, f"processed_docs_batch_{batch_index}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(batch_docs, f)

if __name__ == '__main__':

    tokenizer_name = "naver/splade-v3"
    output_dir = "/group/jmearlesgrp/data/KILT/processed_docs"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 100000
    total_docs = len(RAW_COLLECTION['train'])
    num_batches = (total_docs + batch_size - 1) // batch_size


    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, total_docs)
        batch_raw_docs = RAW_COLLECTION['train'].select(range(start_index, end_index))
        
        with mp.Pool(initializer=init_worker, initargs=(tokenizer_name,)) as pool:
            results = list(tqdm(pool.map(process_doc, batch_raw_docs), total=len(batch_raw_docs)))
        
        batch_processed = [doc for sublist in results for doc in sublist]
        
        save_batch(batch_processed, batch_index, output_dir)
        print(f"Saved batch {batch_index} with {len(batch_processed)} documents.")

        del results, batch_processed, batch_raw_docs

    combined_docs = []
    for batch_index in range(num_batches):
        batch_path = os.path.join(output_dir, f"processed_docs_batch_{batch_index}.pkl")
        with open(batch_path, "rb") as f:
            batch_docs = pickle.load(f)
            combined_docs.extend(batch_docs)
    combined_output_path = os.path.join(output_dir, "combined_processed_docs.pkl")
    with open(combined_output_path, "wb") as f:
        pickle.dump(combined_docs, f)
    print("All batches processed and combined.")
