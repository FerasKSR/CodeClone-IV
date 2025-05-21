import numpy as np
import faiss
import os
from tqdm import tqdm

path1 = "/Users/ismail/Documents/python_projects/CodeClone-IV/data/original_files_embedding1/100005756.npy"
# Load vectors from .npy file
print(f"default data type is {np.load(path1).dtype}") #float64
embedding_1_vectors = np.load(path1).astype('float32')  # FAISS requires float32

print(f"Number of vectors: {embedding_1_vectors.shape[0]}")

path2 = "/Users/ismail/Documents/python_projects/CodeClone-IV/data/original_files_embedding2/100005756.npy"
# Load vectors from .npy file
embedding_2_vectors = np.load(path2).astype('float32')  # FAISS requires float32

print(f"Number of vectors: {embedding_2_vectors.shape[0]}")

def build_faiss_index(folder_path: str, index_name: str, cosine_similarity: bool = False):

    # Collect all .npy file paths
    npy_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npy")]
    npy_files.sort()

    # Load and concatenate vectors
    all_vectors = []
    print(f"Loading {len(npy_files)} files from '{folder_path}'...")
    
    for file_path in tqdm(npy_files):
        vectors = np.load(file_path).astype(np.float32)
        all_vectors.append(vectors)

    # Stack all vectors into one array
    all_vectors = np.vstack(all_vectors)
    print(f"Total vectors: {all_vectors.shape[0]}, Dimension: {all_vectors.shape[1]}")

    if cosine_similarity:
        faiss.normalize_L2(all_vectors)
    # Build FAISS L2 index
    dim = all_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_vectors)

    # Save the index
    faiss.write_index(index, index_name)
    print(f"FAISS index saved as '{index_name}'")


# Example usage:
build_faiss_index("data/original_files_embedding1/", "files_1_l2.index", False)
build_faiss_index("data/original_files_embedding1/", "files_1_cosine.index", True)

build_faiss_index("data/original_files_embedding2/", "files_2_l2.index", False)
build_faiss_index("data/original_files_embedding2/", "files_2_cosine.index", True)
