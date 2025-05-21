import os
import numpy as np
import faiss
from tqdm import tqdm
import psutil
import time
import json

def evaluate_faiss_index(index_path, query_folder, output_file, top_k=5, cosine_similarity=False):
    # === Resource Monitoring Setup ===
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    start_cpu = process.cpu_percent(interval=None)
    start_time = time.time()


    print(f"\nLoading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    dim = index.d
    print(f"FAISS index loaded. Dimension: {dim}")

    # === Get query files ===
    query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.endswith(".npy")]
    query_files.sort()
    total_queries = len(query_files)

    if total_queries == 0:
        print("No .npy query files found in the provided folder.")
        return

    print(f"\nLoading {total_queries} query vectors from: {query_folder}")
    all_vectors = []
    for file_path in tqdm(query_files):
        vectors = np.load(file_path).astype(np.float32)
        all_vectors.append(vectors)

    all_vectors = np.vstack(all_vectors)
    print(f"All vectors shape: {all_vectors.shape}")

    if cosine_similarity:
        faiss.normalize_L2(all_vectors)

    print("\nRunning FAISS search...")
    distances, indices = index.search(all_vectors, top_k)

    # === Evaluation Metrics ===
    correct_matches = 0
    all_results = []

    for query_idx, (D, I) in enumerate(zip(distances, indices)):
        is_correct = bool(I[0] == query_idx)
        if is_correct:
            correct_matches += 1

        all_results.append({
            "file": query_files[query_idx],
            "query_idx": query_idx,
            "top_k_indices": I.tolist(),
            "is_correct": is_correct
        })

    # === Resource Monitoring Summary ===
    end_time = time.time()

    total_time = end_time - start_time
    end_mem = process.memory_info().rss
    final_cpu = process.cpu_percent(interval=0.1)
    total_mem_mb = (end_mem - start_mem) / (1024 * 1024)

    # === Metrics Summary ===
    accuracy = correct_matches / total_queries

    print("\n=== Summary ===")
    print(f"Total queries: {total_queries}")
    print(f"Correct top-1 matches: {correct_matches}")
    print(f"Top-1 Accuracy: {accuracy:.4f}")
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"RAM increased by: {total_mem_mb:.2f} MB")
    print(f"Final CPU usage: {final_cpu:.2f}%")

    # === Save results to JSON ===
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed results to {output_file}")


# === Example Usage ===
if __name__ == "__main__":
    index_path = "files_1_cosine.index"
    query_folder = "/Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding1/"
    output_file = "faiss_query_accuracy1_cosine.json"
    evaluate_faiss_index(index_path, query_folder, output_file, 5, True)

    index_path = "files_1_l2.index"
    query_folder = "/Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding1/"
    output_file = "faiss_query_accuracy1_l2.json"
    evaluate_faiss_index(index_path, query_folder, output_file, 5, False)

    index_path = "files_2_cosine.index"
    query_folder = "/Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding2/"
    output_file = "faiss_query_accuracy1_cosine.json"
    evaluate_faiss_index(index_path, query_folder, output_file, 5, True)

    index_path = "files_2_l2.index"
    query_folder = "/Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding2/"
    output_file = "faiss_query_accuracy1_l2.json"
    evaluate_faiss_index(index_path, query_folder, output_file, 5, False)
    """
    Loading FAISS index from: files_1_cosine.index
    FAISS index loaded. Dimension: 1536

    Loading 80796 query vectors from: /Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding1/
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80796/80796 [00:07<00:00, 11443.15it/s]
    All vectors shape: (80796, 1536)

    Running FAISS search...

    === Summary ===
    Total queries: 80796
    Correct top-1 matches: 54059
    Top-1 Accuracy: 0.6691
    Time taken: 31.29 seconds
    RAM increased by: 1114.94 MB
    Final CPU usage: 0.10%
    Saved detailed results to faiss_query_accuracy1_cosine.json

    Loading FAISS index from: files_1_l2.index
    FAISS index loaded. Dimension: 1536

    Loading 80796 query vectors from: /Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding1/
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80796/80796 [00:07<00:00, 11216.65it/s]
    All vectors shape: (80796, 1536)

    Running FAISS search...

    === Summary ===
    Total queries: 80796
    Correct top-1 matches: 54059
    Top-1 Accuracy: 0.6691
    Time taken: 33.07 seconds
    RAM increased by: 1039.78 MB
    Final CPU usage: 0.00%
    Saved detailed results to faiss_query_accuracy1_l2.json

    Loading FAISS index from: files_2_cosine.index
    FAISS index loaded. Dimension: 3072

    Loading 80796 query vectors from: /Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding2/
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80796/80796 [00:12<00:00, 6534.58it/s]
    All vectors shape: (80796, 3072)

    Running FAISS search...

    === Summary ===
    Total queries: 80796
    Correct top-1 matches: 55751
    Top-1 Accuracy: 0.6900
    Time taken: 62.27 seconds
    RAM increased by: 1663.41 MB
    Final CPU usage: 0.10%
    Saved detailed results to faiss_query_accuracy1_cosine.json

    Loading FAISS index from: files_2_l2.index
    FAISS index loaded. Dimension: 3072

    Loading 80796 query vectors from: /Users/ismail/Documents/python_projects/CodeClone-IV/data/queries_embedding2/
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80796/80796 [00:10<00:00, 7515.72it/s]
    All vectors shape: (80796, 3072)

    Running FAISS search...

    === Summary ===
    Total queries: 80796
    Correct top-1 matches: 55751
    Top-1 Accuracy: 0.6900
    Time taken: 62.19 seconds
    RAM increased by: 2063.27 MB
    Final CPU usage: 0.00%
    Saved detailed results to faiss_query_accuracy1_l2.json"""