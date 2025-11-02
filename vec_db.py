from typing import Dict, List, Annotated
import numpy as np
import os
import gc
import math
import heapq
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
        self.subvector_dim = self.dim // self.n_subvectors
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)

            if os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)

            self.n_patterns = min(256, db_size)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
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

    def get_all_rows(self) -> np.memmap:
        num_records = self._get_num_records()
        return np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
    
    def _find_nearest_clusters(self, query, centroids, n_probes):
        similarities = np.array([self._cal_score(c, query) for c in centroids])
        top_clusters = np.argsort(similarities)[-n_probes:][::-1]
        return top_clusters.tolist()

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
            base = math.sqrt(n_vectors) * 1.5
            n_clusters = int(base)  
            n_clusters = max(256, min(n_clusters, 4096))
            n_clusters = 2 ** math.ceil(math.log2(n_clusters))
            return min(n_clusters, n_vectors)
    
    def _ivf_execute(self, vectors, min_batch_size=10000):
        num_records = self._get_num_records()
        n_clusters = self._determine_n_clusters(num_records)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min_batch_size,
            random_state=DB_SEED_NUMBER,
            max_iter=100,
            n_init=3
        )
        
        batch_size = 50000
        for batch_start in range(0, num_records, batch_size):
            batch_end = min(batch_start + batch_size, num_records)
            batch_vectors = vectors[batch_start:batch_end]
            kmeans.partial_fit(batch_vectors)
        
        centroids = kmeans.cluster_centers_
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
        centroids = centroids.astype(np.float16)
        
        inverted_index = {i: [] for i in range(n_clusters)}
        
        for batch_start in range(0, num_records, batch_size):
            batch_end = min(batch_start + batch_size, num_records)
            batch_vectors = vectors[batch_start:batch_end]
            batch_labels = kmeans.predict(batch_vectors)
            
            for i, label in enumerate(batch_labels):
                inverted_index[label].append(batch_start + i)
        
        return inverted_index, centroids

    def _build_pq_codebooks(self, vectors):
        pq_codebooks = []
        num_records = self._get_num_records()
        
        for subspace in range(self.n_subvectors):
            start_idx = subspace * self.subvector_dim
            end_idx = (subspace + 1) * self.subvector_dim
            
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_patterns,
                batch_size=10000,
                random_state=DB_SEED_NUMBER + subspace,
                max_iter=100,
                n_init=3
            )
            
            batch_size = 50000
            for batch_start in range(0, num_records, batch_size):
                batch_end = min(batch_start + batch_size, num_records)
                batch_vectors = vectors[batch_start:batch_end]
                subspace_vectors = batch_vectors[:, start_idx:end_idx]
                kmeans.partial_fit(subspace_vectors)
            
            pq_codebooks.append(kmeans.cluster_centers_)
        
        for i in range(len(pq_codebooks)):
            pq_codebooks[i] /= (np.linalg.norm(pq_codebooks[i], axis=1, keepdims=True) + 1e-10)
        
        return pq_codebooks

    def _compute_compressed_codes(self, vectors, pq_codebooks, inverted_index):
        num_records = self._get_num_records()
        os.makedirs(self.index_path, exist_ok=True)
        index_to_cluster = {}
        for cluster_id, indices in inverted_index.items():
            for idx in indices:
                index_to_cluster[idx] = cluster_id

        cluster_codes = {}
        for cluster_id in inverted_index.keys():
            if inverted_index[cluster_id]:
                n_vectors_in_cluster = len(inverted_index[cluster_id])
                cluster_codes_path = os.path.join(self.index_path, f"index_codes_{cluster_id}.dat")
                cluster_codes[cluster_id] = np.memmap(
                    cluster_codes_path, dtype=np.uint8, mode='w+',
                    shape=(n_vectors_in_cluster, self.n_subvectors)
                )

        cluster_positions = {cluster_id: 0 for cluster_id in inverted_index.keys()}

        batch_size = 10000
        for batch_start in range(0, num_records, batch_size):
            batch_end = min(batch_start + batch_size, num_records)
            batch_vectors = vectors[batch_start:batch_end]

            batch_codes = np.zeros((len(batch_vectors), self.n_subvectors), dtype=np.uint8)
            for subspace in range(self.n_subvectors):
                start_dim = subspace * self.subvector_dim
                end_dim = (subspace + 1) * self.subvector_dim
                query_chunk = batch_vectors[:, start_dim:end_dim]
                codebook_subspace = pq_codebooks[subspace]

                distances = np.linalg.norm(
                    query_chunk[:, np.newaxis, :] - codebook_subspace[np.newaxis, :, :],
                    axis=2
                )
                batch_codes[:, subspace] = np.argmin(distances, axis=1)

            for i in range(len(batch_vectors)):
                global_idx = batch_start + i
                cluster_id = index_to_cluster.get(global_idx)
                if cluster_id is not None:
                    pos = cluster_positions[cluster_id]
                    cluster_codes[cluster_id][pos] = batch_codes[i]
                    cluster_positions[cluster_id] += 1

        for codes_memmap in cluster_codes.values():
            codes_memmap.flush()
            del codes_memmap

        gc.collect()

    def _build_index(self):
        vectors = self.get_all_rows()
        inverted_index, centroids = self._ivf_execute(vectors)

        pq_codebooks = self._build_pq_codebooks(vectors)
        self._compute_compressed_codes(vectors, pq_codebooks, inverted_index)

        current_offset = 0

        for cluster_id in range(len(centroids)):
            indices = inverted_index.get(cluster_id, [])
            count = len(indices)
            current_offset += count

        os.makedirs(self.index_path, exist_ok=True)

        centroids_path = os.path.join(self.index_path, "centroids.dat")

        mmap_centroids = np.memmap(centroids_path, dtype=np.float16, mode='w+', shape=centroids.shape)
        mmap_centroids[:] = centroids[:]
        mmap_centroids.flush()
        del mmap_centroids

        for i, cb in enumerate(pq_codebooks):
            cb_path = os.path.join(self.index_path, f"pq_cb_{i}.dat")
            mmap_cb = np.memmap(cb_path, dtype=np.float16, mode='w+', shape=cb.shape)
            mmap_cb[:] = cb[:]
            mmap_cb.flush()
            del mmap_cb

        max_uint16 = np.iinfo(np.uint16).max
        for cluster_id, indices in inverted_index.items():
            indices = np.array(indices, dtype=np.uint32)
            diffs = np.diff(indices, prepend=0)
            
            if diffs.max() <= max_uint16:
                deltas = diffs.astype(np.uint16)
            else:
                deltas = diffs.astype(np.uint32)
            
            path = os.path.join(self.index_path, f"deltas_{cluster_id}.dat")
            with open(path, 'wb') as f:
                np.savez_compressed(f, deltas=deltas)

        del vectors, centroids, pq_codebooks
        gc.collect()

    def _load_index(self):
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        centroids = np.memmap(centroids_path, dtype=np.float16, mode='r').reshape(-1, DIMENSION)

        pq_codebooks = []
        i = 0
        while True:
            cb_path = os.path.join(self.index_path, f"pq_cb_{i}.dat")
            if not os.path.exists(cb_path):
                break
            mmap_cb = np.memmap(cb_path, dtype=np.float16, mode='r')
            n_codewords = int(len(mmap_cb) / self.subvector_dim)
            pq_codebooks.append(mmap_cb.reshape(n_codewords, self.subvector_dim))
            i += 1

        return centroids, pq_codebooks

    def _load_cluster_codes(self, cluster_id):
        codes_path = os.path.join(self.index_path, f"index_codes_{cluster_id}.dat")
        if os.path.exists(codes_path):
            return np.fromfile(codes_path, dtype=np.uint8).reshape(-1, self.n_subvectors)
        return None
    
    def _load_cluster_indices(self, cluster_id):
        indices_path = os.path.join(self.index_path, f"deltas_{cluster_id}.dat")
        if os.path.exists(indices_path):
            with open(indices_path, 'rb') as f:
                data = np.load(f)['deltas']
                return data.cumsum().astype(np.int32).tolist()
        return None

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, n_probes=None):
        query = np.array(query, dtype=np.float32).flatten()
        query /= np.linalg.norm(query) + 1e-10

        centroids, pq_codebooks = self._load_index()
        n_clusters = centroids.shape[0]

        if n_probes is None:
            n_probes = max(12, min(80, n_clusters // 6))

        cluster_ids = self._find_nearest_clusters(query, centroids, n_probes)

        query_chunks = [
            query[i * self.subvector_dim:(i + 1) * self.subvector_dim].astype(np.float16)
            for i in range(self.n_subvectors)
        ]

        top_heap = []
        factor = min(250, max(75, n_clusters // 4))
        n_take = top_k * factor

        for cid in cluster_ids:
            indices = self._load_cluster_indices(cid)

            if indices is None or len(indices) == 0:
                continue

            local_codes = self._load_cluster_codes(cid)

            dist = np.zeros(len(indices), dtype=np.float32)
            for subspace in range(self.n_subvectors):
                q_chunk = query_chunks[subspace]
                codebook_subspace = pq_codebooks[subspace]

                codes = local_codes[:, subspace]
                patterns = codebook_subspace[codes]
                diff = patterns - q_chunk
                dist += np.einsum("ij,ij->i", diff, diff)

            for d, idx in zip(dist, indices):
                if len(top_heap) < n_take:
                    heapq.heappush(top_heap, (-d, idx))
                elif d < -top_heap[0][0]:
                    heapq.heapreplace(top_heap, (-d, idx))

            del local_codes, dist

        if not top_heap:
            return []

        top_candidates = np.array([idx for _, idx in top_heap], dtype=np.int32)
        del top_heap, query_chunks, pq_codebooks

        exact_sims = np.fromiter(
            (self._cal_score(self.get_one_row(idx), query) for idx in top_candidates),
            dtype=np.float32,
            count=len(top_candidates),
        )

        reranked_idx = np.argsort(exact_sims)[-top_k:][::-1]
        final_results = top_candidates[reranked_idx].tolist()

        return final_results
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity