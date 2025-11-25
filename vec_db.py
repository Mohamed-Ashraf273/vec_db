from typing import Dict, List, Annotated
import numpy as np
import os
import gc
import math
import heapq
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.dim = DIMENSION
        
        # Index configuration - RESIDUAL quantization for better compression
        self.db_size = db_size if db_size is not None else 0
        self.n_subvectors = 8
        self.subvector_dim = self.dim // self.n_subvectors
        self.n_patterns = 240
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            # if os.path.exists(self.db_path):
            #     os.remove(self.db_path)

            if os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)

            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        # rng = np.random.default_rng(DB_SEED_NUMBER)
        # vectors = rng.random((size, DIMENSION), dtype=np.float32)
        # vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        # self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 64)]):
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
    
    def _find_nearest_clusters(self, query, n_clusters, n_probes):
        similarities = []
        for i in range(n_clusters):
            centroid = self._get_centroid(i)
            sim = self._call_score(centroid, query)
            similarities.append(sim)
            del centroid
        
        similarities = np.array(similarities)
        n_probes = min(n_probes, n_clusters)
        
        if n_probes >= n_clusters:
            top_clusters = np.arange(n_clusters)
        else:
            top_clusters = np.argpartition(similarities, -n_probes)[-n_probes:]
            top_clusters = top_clusters[np.argsort(similarities[top_clusters])[::-1]]
        
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
            base = math.sqrt(n_vectors)
            n_clusters = int(base)  
            n_clusters = max(256, min(n_clusters, 1024))
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

    def _build_pq_codebooks(self, vectors, centroids, inverted_index):
        pq_codebooks = []
        num_records = self._get_num_records()
        
        index_to_centroid = np.zeros((num_records, self.dim), dtype=np.float16)
        for cluster_id, indices in inverted_index.items():
            for idx in indices:
                index_to_centroid[idx] = centroids[cluster_id]
        
        for subspace in range(self.n_subvectors):
            start_idx = subspace * self.subvector_dim
            end_idx = (subspace + 1) * self.subvector_dim
            
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_patterns,
                batch_size=10000,
                random_state=DB_SEED_NUMBER + subspace,
                max_iter=300,
                n_init=5
            )
            
            batch_size = 50000
            for batch_start in range(0, num_records, batch_size):
                batch_end = min(batch_start + batch_size, num_records)
                batch_vectors = vectors[batch_start:batch_end]
                batch_centroids = index_to_centroid[batch_start:batch_end]
                
                residuals = batch_vectors - batch_centroids
                subspace_residuals = residuals[:, start_idx:end_idx]
                kmeans.partial_fit(subspace_residuals)
            
            pq_codebooks.append(kmeans.cluster_centers_)
        
        for i in range(len(pq_codebooks)):
            pq_codebooks[i] /= (np.linalg.norm(pq_codebooks[i], axis=1, keepdims=True) + 1e-10)
        
        del index_to_centroid
        return pq_codebooks

    def _compute_compressed_codes(self, vectors, pq_codebooks, inverted_index, centroids):
        num_records = self._get_num_records()
        index_to_cluster = {}
        for cluster_id, indices in inverted_index.items():
            for idx in indices:
                index_to_cluster[idx] = cluster_id

        cluster_codes = {}
        batch_size = 10000
        for batch_start in range(0, num_records, batch_size):
            batch_end = min(batch_start + batch_size, num_records)
            batch_vectors = vectors[batch_start:batch_end]

            batch_residuals = np.zeros_like(batch_vectors)
            for i in range(len(batch_vectors)):
                global_idx = batch_start + i
                cluster_id = index_to_cluster.get(global_idx)
                if cluster_id is not None:
                    batch_residuals[i] = batch_vectors[i] - centroids[cluster_id]

            batch_codes = np.zeros((len(batch_vectors), self.n_subvectors), dtype=np.uint8)
            for subspace in range(self.n_subvectors):
                start_dim = subspace * self.subvector_dim
                end_dim = (subspace + 1) * self.subvector_dim
                residual_chunk = batch_residuals[:, start_dim:end_dim]
                codebook_subspace = pq_codebooks[subspace]

                distances = self._call_score(residual_chunk, codebook_subspace.T)
                batch_codes[:, subspace] = np.argmax(distances, axis=1)

            for i in range(len(batch_vectors)):
                global_idx = batch_start + i
                cluster_id = index_to_cluster.get(global_idx)
                if cluster_id is not None:
                    if cluster_id not in cluster_codes:
                        cluster_codes[cluster_id] = []
                    cluster_codes[cluster_id].append(batch_codes[i])

        for cluster_id, codes_list in cluster_codes.items():
            if codes_list:
                codes_array = np.array(codes_list, dtype=np.uint8)
                codes_path = os.path.join(self.index_path, f"index_codes_{cluster_id}.dat")
                mmap_codes = np.memmap(codes_path, dtype=np.uint8, mode='w+', shape=codes_array.shape)
                mmap_codes[:] = codes_array[:]
                mmap_codes.flush()
                del mmap_codes
        
        del cluster_codes
        gc.collect()

    def _build_index(self):
        os.makedirs(self.index_path, exist_ok=True)
        vectors = self.get_all_rows()
        inverted_index, centroids = self._ivf_execute(vectors)
        pq_codebooks = self._build_pq_codebooks(vectors, centroids, inverted_index)
        self._compute_compressed_codes(vectors, pq_codebooks, inverted_index, centroids)
        
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        mmap_centroids = np.memmap(centroids_path, dtype=np.float16, mode='w+', shape=centroids.shape)
        mmap_centroids[:] = centroids[:]
        mmap_centroids.flush()
        del mmap_centroids

        pq_codebooks_array = np.stack(pq_codebooks)
        cb_path = os.path.join(self.index_path, "pq_cb.dat")
        mmap_cb = np.memmap(cb_path, dtype=np.float16, mode='w+', shape=pq_codebooks_array.shape)
        mmap_cb[:] = pq_codebooks_array[:]
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
            deltas.tofile(path)

        del vectors, centroids, pq_codebooks
        gc.collect()

    def _get_centroid(self, idx):
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        offset = idx * DIMENSION * np.dtype(np.float16).itemsize
        mmap_centroid = np.memmap(centroids_path, dtype=np.float16, mode='r', 
                                  shape=(DIMENSION,), offset=offset)
        return np.array(mmap_centroid, dtype=np.float16)

    def _get_n_clusters(self):
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        file_size = os.path.getsize(centroids_path)
        return file_size // (DIMENSION * np.dtype(np.float16).itemsize)
    
    def _get_pq_codebooks(self):
        cb_path = os.path.join(self.index_path, "pq_cb.dat")
        if not os.path.exists(cb_path):
            return None
        
        all_codebooks = np.memmap(cb_path, dtype=np.float16, mode='r', 
                                shape=(self.n_subvectors, self.n_patterns, self.subvector_dim))
        
        return np.array(all_codebooks, dtype=np.float16)

    def _get_cluster_codes(self, cluster_id):
        codes_path = os.path.join(self.index_path, f"index_codes_{cluster_id}.dat")
        if not os.path.exists(codes_path):
            return None
        
        file_size = os.path.getsize(codes_path)
        n_rows = file_size // (self.n_subvectors * np.dtype(np.uint8).itemsize)
        mmap_codes = np.memmap(codes_path, dtype=np.uint8, mode='r', shape=(n_rows, self.n_subvectors))
        return np.array(mmap_codes, dtype=np.uint8)
        
    def _get_cluster_indices(self, cluster_id):
        indices_path = os.path.join(self.index_path, f"deltas_{cluster_id}.dat")
        if os.path.exists(indices_path):
            deltas = np.fromfile(indices_path, dtype=np.uint16)
            indices = deltas.cumsum().astype(np.int32)
            num_records = self._get_num_records()
            valid_mask = (indices >= 0) & (indices < num_records)
            return indices[valid_mask]
        return None
        
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, n_probes=None):
        query = np.array(query, dtype=np.float32).flatten()
        query /= np.linalg.norm(query) + 1e-10

        n_clusters = self._get_n_clusters()

        if n_probes is None:
            n_probes = max(12, min(80, n_clusters // 6))

        cluster_ids = self._find_nearest_clusters(query, n_clusters, n_probes)

        factor = min(160, max(100, n_clusters // 4)) * 3
        n_take = top_k * factor

        candidate_heap = []

        codebooks = self._get_pq_codebooks()

        for cid in cluster_ids:
            indices = self._get_cluster_indices(cid)
            if indices is None or len(indices) == 0:
                continue

            local_codes = self._get_cluster_codes(cid)
            if local_codes is None:
                del indices
                continue

            centroid = self._get_centroid(cid)
            query_residual = query - centroid

            n_vecs = min(len(local_codes), len(indices))
            local_codes = local_codes[:n_vecs]
            indices = indices[:n_vecs]

            dist = np.zeros(n_vecs, dtype=np.float16)

            subspace = np.empty(self.n_patterns, dtype=np.float32)

            for i in range(self.n_subvectors):
                start_idx = i * self.subvector_dim
                end_idx = (i + 1) * self.subvector_dim

                qr = query_residual[start_idx:end_idx]
                cb = codebooks[i]
                if cb is None:
                    continue

                cb = np.asarray(cb, dtype=np.float32)

                q_norm = float(self._call_score(qr, qr))
                cb_norm = np.sum(cb * cb, axis=1)
                dotp = cb @ qr

                subspace[:] = q_norm + cb_norm - 2 * dotp
                dist += subspace[local_codes[:, i]].astype(np.float16)

            if len(candidate_heap) < n_take:
                needed = n_take - len(candidate_heap)
                take_n = min(needed, n_vecs)
                top_idxs = np.argpartition(dist, take_n - 1)[:take_n]
                for j in top_idxs:
                    heapq.heappush(candidate_heap, (-float(dist[j]), int(indices[j])))
            else:
                threshold = -candidate_heap[0][0]
                mask = dist < threshold
                for idx_pos in np.where(mask)[0]:
                    d = float(dist[idx_pos])
                    if d < threshold:
                        heapq.heapreplace(candidate_heap, (-d, int(indices[idx_pos])))
                        threshold = -candidate_heap[0][0]

        gc.collect()

        if not candidate_heap:
            return []

        seen = set()
        unique_candidates = []
        for _, idx in candidate_heap:
            if idx not in seen:
                seen.add(idx)
                unique_candidates.append(idx)

        n_keep = int(len(unique_candidates) * 0.9)
        candidate_ids = unique_candidates[:n_keep]

        del candidate_heap

        final_heap = []
        
        for idx in candidate_ids:
            vec = self.get_one_row(idx)
            sim = self._call_score(vec, query)
            if len(final_heap) < top_k:
                heapq.heappush(final_heap, (float(sim), idx))
            elif sim > final_heap[0][0]:
                heapq.heapreplace(final_heap, (float(sim), idx))

        del candidate_ids
        gc.collect()

        final_results = [idx for _, idx in heapq.nlargest(top_k, final_heap)]
        return final_results

    def _call_score(self, vec1, vec2):
        return np.dot(vec1, vec2)