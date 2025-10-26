from typing import Dict, List, Annotated
import numpy as np
import os
import math
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.dim = DIMENSION
        
        # Index configuration - these are fixed-size values, allowed in init
        self.db_size = db_size if db_size is not None else 0
        self.n_subvectors = 7
        self.n_patterns = 256
        self.subvector_dim = self.dim // self.n_subvectors
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)

            self.n_patterns = min(self.n_patterns, db_size)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        # Rebuild index for simplicity (insert not required)
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def _determine_n_clusters(self, n_vectors):
        # Resources:
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        # https://github.com/facebookresearch/faiss/wiki/The-index-factory#ivf-indexes
        # https://www-users.cse.umn.edu/~kumar001/papers/high_dim_clustering_19.pdf
        # Heuristic: n_clusters = 2 ^ ceil(log2(sqrt(n_vectors))) bounded between 256 and 4096
        if n_vectors < 256:
            n_clusters = max(1, n_vectors // 4)
            n_clusters = 2 ** math.ceil(math.log2(n_clusters))
            return max(1, n_clusters)
        else:
            base = math.sqrt(n_vectors)
            n_clusters = int(base)  
            n_clusters = max(256, min(n_clusters, 4096))
            n_clusters = 2 ** math.ceil(math.log2(n_clusters))
            return min(n_clusters, n_vectors)
    
    def _ivf_execute(self, vectors, min_batch_size=10000, max_batch_size=50000) -> dict:
        n_vectors = vectors.shape[0]
        n_clusters = self._determine_n_clusters(n_vectors)
        raw_batch_size = 50 * n_clusters
        batch_size = max(min_batch_size, min(raw_batch_size, max_batch_size))
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, 
                                random_state=42, max_iter=100, n_init=3)
        labels = kmeans.fit_predict(vectors)
        centroids = kmeans.cluster_centers_

        inverted_index = {}
        for i, label in enumerate(labels):
            if label not in inverted_index:
                inverted_index[label] = []
            inverted_index[label].append(i)
        
        return inverted_index, centroids
    
    def _build_pq_codebooks(self, vectors):
        pq_codebooks = []
        for subspace in range(self.n_subvectors):
            start_idx = subspace * self.subvector_dim
            end_idx = (subspace + 1) * self.subvector_dim
            subspace_vectors = vectors[:, start_idx:end_idx]
            
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_patterns,
                batch_size=10000,
                random_state=42 + subspace,
                max_iter=100,
                n_init=3
            )
            kmeans.fit(subspace_vectors)
            pq_codebooks.append(kmeans.cluster_centers_)
        
        return pq_codebooks
    
    def _compute_compressed_codes(self, vectors, pq_codebooks):
        n_vectors = vectors.shape[0]
        compressed_codes = np.memmap(
            self.index_path,
            dtype=np.uint8,
            mode='w+',
            shape=(n_vectors, self.n_subvectors)
        )

        for subspace in range(self.n_subvectors):
            start_idx = subspace * self.subvector_dim
            end_idx = (subspace + 1) * self.subvector_dim
            subspace_vectors = vectors[:, start_idx:end_idx]
            codebook = pq_codebooks[subspace]
            
            batch_size = 10000
            for batch_start in range(0, n_vectors, batch_size):
                batch_end = min(batch_start + batch_size, n_vectors)
                batch_vectors = subspace_vectors[batch_start:batch_end]
                
                batch_distances = np.linalg.norm(
                    batch_vectors[:, np.newaxis, :] - codebook[np.newaxis, :, :], 
                    axis=2
                )
                compressed_codes[batch_start:batch_end, subspace] = np.argmin(batch_distances, axis=1)

        compressed_codes.flush()
        return compressed_codes
    

    def _build_index(self):
        vectors = self.get_all_rows()
        n_vectors = vectors.shape[0]
        inverted_index, centroids = self._ivf_execute(vectors)
        pq_codebooks = self._build_pq_codebooks(vectors)
        compressed_codes = self._compute_compressed_codes(vectors, pq_codebooks)
        
        index_meta = {
            'inverted_index': inverted_index,
            'centroids': centroids,
            'pq_codebooks': pq_codebooks,
        }
        np.save(self.index_path.split('.')[0] + "_meta.npy", index_meta)

        del vectors
        del compressed_codes

    def _load_index_meta(self):
        index_meta = np.load(self.index_path.split('.')[0] + "_meta.npy", allow_pickle=True).item()
        return (index_meta['inverted_index'], index_meta['centroids'], 
                index_meta['pq_codebooks'])

    def _find_nearest_clusters(self, query, centroids, n_probes):
        similarities = []
        for cluster_id, centroid in enumerate(centroids):
            similarity = self._cal_score(query, centroid)
            similarities.append((cluster_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cluster_id for cluster_id, _ in similarities[:n_probes]]

    def _compute_pq_distance(self, query, compressed_codes, pq_codebooks):
        total_distance = 0.0
        for subspace in range(self.n_subvectors):
            start_idx = subspace * self.subvector_dim
            end_idx = (subspace + 1) * self.subvector_dim
            pattern = pq_codebooks[subspace][compressed_codes[subspace]]
            query_chunk = query[start_idx:end_idx]
            total_distance += np.sum((query_chunk - pattern) ** 2)
        return np.sqrt(total_distance)


    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, n_probes=None):
        query = np.array(query).flatten()
        if len(query) != self.dim:
            raise ValueError(f"Query dimension {len(query)} != database dimension {self.dim}")
        
        inverted_index, centroids, pq_codebooks = self._load_index_meta()
        n_clusters = len(centroids)

        if n_probes is None:
            n_probes = max(4, min(32, n_clusters // 16))
        
        cluster_ids = self._find_nearest_clusters(query, centroids, n_probes)
        
        candidate_indices = []
        for cluster_id in cluster_ids:
            if cluster_id in inverted_index:
                candidate_indices.extend(inverted_index[cluster_id])
        
        compressed_codes = np.memmap(
            self.index_path,
            dtype=np.uint8,
            mode='r',
            shape=(self.db_size, self.n_subvectors)
        )
        
        pq_candidates = []
        for idx in candidate_indices:
            compressed_vector = compressed_codes[idx]
            pq_distance = self._compute_pq_distance(query, compressed_vector, pq_codebooks)
            pq_candidates.append((idx, pq_distance))
        
        pq_candidates.sort(key=lambda x: x[1])
        candidate_factor = max(10, min(50, n_clusters // 20))
        top_pq_candidates = pq_candidates[:top_k * candidate_factor]  # retrieve more for re-ranking

        reranked_results = []
        for idx, pq_distance in top_pq_candidates:
            original_vector = self.get_one_row(idx)
            exact_similarity = self._cal_score(query, original_vector)
            reranked_results.append((idx, exact_similarity))
        
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in reranked_results[:top_k]]

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity