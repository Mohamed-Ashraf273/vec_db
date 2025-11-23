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
        
        # Index configuration - these are fixed-size values, allowed in init
        self.db_size = db_size if db_size is not None else 0
        self.n_subvectors = 8
        self.subvector_dim = self.dim // self.n_subvectors
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            # if os.path.exists(self.db_path):
            #     os.remove(self.db_path)

            if os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)

            self.n_patterns = min(256, db_size)
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
            sim = np.dot(centroid, query)
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
            n_clusters = max(256, min(n_clusters, 2048))
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
        index_to_cluster = {}
        for cluster_id, indices in inverted_index.items():
            for idx in indices:
                index_to_cluster[idx] = cluster_id

        cluster_codes = {}
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
                    if cluster_id not in cluster_codes:
                        cluster_codes[cluster_id] = []
                    cluster_codes[cluster_id].append(batch_codes[i])

        for cluster_id, codes_list in cluster_codes.items():
            if codes_list:
                codes_array = np.array(codes_list, dtype=np.uint8)
                codes_path = os.path.join(self.index_path, f"index_codes_{cluster_id}.dat")
                
                if codes_array.nbytes > 50000:
                    with open(codes_path, 'wb') as f:
                        np.savez_compressed(f, codes=codes_array)
                else:
                    codes_array.tofile(codes_path)
        
        del cluster_codes
        gc.collect()

    def _build_index(self):
        os.makedirs(self.index_path, exist_ok=True)
        vectors = self.get_all_rows()
        inverted_index, centroids = self._ivf_execute(vectors)
        pq_codebooks = self._build_pq_codebooks(vectors)
        self._compute_compressed_codes(vectors, pq_codebooks, inverted_index)
        
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
    
    def _get_pq_codebook(self, subspace_idx):
        cb_path = os.path.join(self.index_path, f"pq_cb_{subspace_idx}.dat")
        if not os.path.exists(cb_path):
            return None
        mmap_cb = np.memmap(cb_path, dtype=np.float16, mode='r')
        n_codewords = int(len(mmap_cb) / self.subvector_dim)
        return mmap_cb.reshape(n_codewords, self.subvector_dim)

    def _load_cluster_codes(self, cluster_id):
        codes_path = os.path.join(self.index_path, f"index_codes_{cluster_id}.dat")
        if os.path.exists(codes_path):
            try:
                with open(codes_path, 'rb') as f:
                    data = np.load(f)['codes']
                    return data
            except:
                return np.fromfile(codes_path, dtype=np.uint8).reshape(-1, self.n_subvectors)
        return None
    
    def _load_cluster_indices(self, cluster_id):
        indices_path = os.path.join(self.index_path, f"deltas_{cluster_id}.dat")
        if os.path.exists(indices_path):
            deltas = np.fromfile(indices_path, dtype=np.uint16)
            indices = deltas.cumsum().astype(np.int32)
            num_records = self._get_num_records()
            valid_mask = (indices >= 0) & (indices < num_records)
            return indices[valid_mask]
        return None

    def _process_batch(self, batch_ids, query, final_heap, top_k):
        if not batch_ids:
            return
        
        num_records = self._get_num_records()
        valid_batch = [idx for idx in batch_ids if 0 <= idx < num_records]
        if not valid_batch:
            return
        
        batch_size = len(valid_batch)
        batch_vectors = np.empty((batch_size, DIMENSION), dtype=np.float32)
        
        for i, idx in enumerate(valid_batch):
            offset = np.int64(idx) * np.int64(DIMENSION) * np.int64(ELEMENT_SIZE)
            vec = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                           shape=(DIMENSION,), offset=offset)
            batch_vectors[i] = vec
        
        similarities = np.dot(batch_vectors, query)
        
        for i, idx in enumerate(valid_batch):
            sim = float(similarities[i])
            if len(final_heap) < top_k:
                heapq.heappush(final_heap, (sim, idx))
            elif sim > final_heap[0][0]:
                heapq.heapreplace(final_heap, (sim, idx))
        
        del batch_vectors, similarities

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, n_probes=None):
        query = np.array(query, dtype=np.float32).flatten()
        query /= np.linalg.norm(query) + 1e-10

        n_clusters = self._get_n_clusters()

        if n_probes is None:
            n_probes = max(16, min(80, n_clusters // 3))

        cluster_ids = self._find_nearest_clusters(query, n_clusters, n_probes)

        subspace_distances = []
        for i in range(self.n_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            q_chunk = query[start_idx:end_idx]
            
            codebook = self._get_pq_codebook(i)
            if codebook is None:
                subspace_distances.append(np.zeros(self.n_patterns, dtype=np.float32))
                continue
            
            codebook_f32 = codebook.astype(np.float32)
            q_norm_sq = np.dot(q_chunk, q_chunk)
            cb_norm_sq = np.sum(codebook_f32 * codebook_f32, axis=1)
            dot_product = np.dot(codebook_f32, q_chunk)
            dists = q_norm_sq + cb_norm_sq - 2 * dot_product
            
            subspace_distances.append(dists)
            del codebook, codebook_f32, cb_norm_sq, dot_product

        n = self._get_num_records()
        b = 18.0 / 19.0
        a = (10**6) * (2 - b)
        C = a / n + b
        factor =  int(min(90, max(30, n_clusters // 12)) * C)
        n_take = top_k * factor

        candidate_heap = []
        
        for cid in cluster_ids:
            indices = self._load_cluster_indices(cid)
            if indices is None or len(indices) == 0:
                continue

            local_codes = self._load_cluster_codes(cid)
            if local_codes is None:
                del indices
                continue

            n_vecs = min(len(local_codes), len(indices))
            local_codes = local_codes[:n_vecs]
            indices = indices[:n_vecs]
            dist = np.sum([subspace_distances[i][local_codes[:, i]] for i in range(self.n_subvectors)], axis=0)
            
            if len(candidate_heap) < n_take:
                n_add = min(n_vecs, n_take - len(candidate_heap))
                for i in range(n_add):
                    heapq.heappush(candidate_heap, (-float(dist[i]), int(indices[i])))
            else:
                threshold = -candidate_heap[0][0]
                better_mask = dist < threshold
                better_indices = np.where(better_mask)[0]
                
                for idx in better_indices:
                    d = float(dist[idx])
                    if d < threshold:
                        heapq.heapreplace(candidate_heap, (-d, int(indices[idx])))
                        threshold = -candidate_heap[0][0]

            del local_codes, dist, indices

        del subspace_distances
        
        if not candidate_heap:
            return []

        seen = set()
        candidate_ids = []
        for _, idx in candidate_heap:
            if idx not in seen:
                seen.add(idx)
                candidate_ids.append(idx)
        del candidate_heap, seen
        
        final_heap = []
        batch_size = 256
        
        for i in range(0, len(candidate_ids), batch_size):
            batch_ids = candidate_ids[i:i+batch_size]
            self._process_batch(batch_ids, query, final_heap, top_k)
        
        del candidate_ids
        
        final_results = [idx for _, idx in heapq.nlargest(top_k, final_heap)]
        
        return final_results
    
    def _cal_score(self, vec1, vec2):
        return np.dot(vec1, vec2)