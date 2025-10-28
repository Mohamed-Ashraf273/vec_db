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
            offset = np.int64(row_num) * np.int64(DIMENSION) * np.int64(ELEMENT_SIZE)
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
    
    def _ivf_execute(self, vectors, min_batch_size=10000, max_batch_size=50000):
        n_vectors = vectors.shape[0]
        self.n_clusters = self._determine_n_clusters(n_vectors)
        
        raw_batch_size = 50 * self.n_clusters
        batch_size = max(min_batch_size, min(raw_batch_size, max_batch_size))

        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size,
                                random_state=DB_SEED_NUMBER, max_iter=100, n_init=3)
        labels = kmeans.fit_predict(vectors)
        centroids = kmeans.cluster_centers_

        clusters = []
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            clusters.append(cluster_indices)
        
        total_cluster_data = self.n_clusters + sum(len(cluster) for cluster in clusters) + self.n_clusters
        centroids_size = self.n_clusters * self.dim
        total_size = total_cluster_data + centroids_size
        
        index_mmap = np.memmap(self.index_path, dtype=np.float32, mode='w+', shape=(total_size,))
        
        header_size = self.n_clusters
        cluster_offsets = np.zeros(self.n_clusters, dtype=np.int32)
        
        current_pos = header_size
        
        for cluster_id, cluster_indices in enumerate(clusters):
            cluster_offsets[cluster_id] = current_pos
            index_mmap[current_pos] = len(cluster_indices)
            index_mmap[current_pos + 1: current_pos + 1 + len(cluster_indices)] = cluster_indices
            current_pos += 1 + len(cluster_indices)
        
        index_mmap[0:header_size] = cluster_offsets
        
        centroids_start = current_pos
        centroids_flat = centroids.flatten()
        index_mmap[centroids_start:centroids_start + centroids_size] = centroids_flat
        
        index_mmap.flush()

    def _build_index(self):
        vectors = self.get_all_rows()
        self._ivf_execute(vectors)

    def _find_nearest_clusters(self, query, centroids, n_probes):
        similarities = []
        for cluster_id, centroid in enumerate(centroids):
            similarity = self._cal_score(query, centroid)
            similarities.append((cluster_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cluster_id for cluster_id, _ in similarities[:n_probes]]
    
    def _load_cluster_indices(self, cluster_id):
        header_mmap = np.memmap(self.index_path, dtype=np.float32, mode='r', shape=(self.n_clusters,))
        cluster_offset = int(header_mmap[cluster_id])
        
        full_mmap = np.memmap(self.index_path, dtype=np.float32, mode='r')
        cluster_size = int(full_mmap[cluster_offset])
        cluster_indices = full_mmap[cluster_offset + 1: cluster_offset + 1 + cluster_size]
        
        return cluster_indices.astype(np.int32).copy()

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, n_probes=None):
        query = np.array(query).flatten()
        if len(query) != self.dim:
            raise ValueError(f"Query dimension {len(query)} != database dimension {self.dim}")

        index_mmap = np.memmap(self.index_path, dtype=np.float32, mode='r')
        
        centroids_size = self.n_clusters * self.dim
        centroids_start = len(index_mmap) - centroids_size
        centroids_flat = index_mmap[centroids_start:]
        centroids = centroids_flat.reshape(self.n_clusters, self.dim)

        if n_probes is None:
            n_probes = max(4, min(32, self.n_clusters // 16))
        
        cluster_ids = self._find_nearest_clusters(query, centroids, n_probes)

        candidate_indices = []
        for cluster_id in cluster_ids:
            cluster_indices = self._load_cluster_indices(cluster_id)
            candidate_indices.extend(cluster_indices)
        
        candidate_indices = list(set(candidate_indices))
        
        results = []
        for idx in candidate_indices:
            vector = self.get_one_row(idx)
            score = self._cal_score(query, vector)
            results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in results[:top_k]]

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity