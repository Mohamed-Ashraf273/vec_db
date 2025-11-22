from typing import Dict, List, Annotated
import numpy as np
import os
import gc
import math
import heapq
from sklearn.cluster import MiniBatchKMeans
from tree_node import TreeNode

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
        
    def _build_tree(self, vectors, indices, depth, max_depth, min_leaf_size=1000):
        if depth == max_depth or len(indices) <= min_leaf_size:
            return TreeNode(indices=np.array(indices))

        subset = vectors[indices]

        mean = np.mean(subset, axis=0)
        centered = subset - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]

        proj = subset @ direction

        threshold = np.median(proj)

        left_mask = proj <= threshold
        right_mask = proj > threshold

        left_indices = np.array(indices)[left_mask]
        right_indices = np.array(indices)[right_mask]

        left_child = self._build_tree(vectors, left_indices, depth+1, max_depth, min_leaf_size)
        right_child = self._build_tree(vectors, right_indices, depth+1, max_depth, min_leaf_size)

        return TreeNode(
            direction=direction,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

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

    def _compute_compressed_codes(self, vectors, pq_codebooks, leaf_index):
        num_records = self._get_num_records()

        leaf_codes_memmap = {}

        for leaf_id, idx_list in leaf_index.items():
            n_vecs = len(idx_list)
            path = os.path.join(self.index_path, f"index_codes_{leaf_id}.dat")
            leaf_codes_memmap[leaf_id] = np.memmap(path, dtype=np.uint8, mode='w+', shape=(n_vecs, self.n_subvectors))

        batch_size = 10000

        for batch_start in range(0, num_records, batch_size):
            batch_end = min(batch_start + batch_size, num_records)
            batch_vectors = vectors[batch_start:batch_end]

            batch_codes = np.zeros((len(batch_vectors), self.n_subvectors), dtype=np.uint8)

            for s in range(self.n_subvectors):
                start = s * self.subvector_dim
                end = (s + 1) * self.subvector_dim

                q = batch_vectors[:, start:end]
                cb = pq_codebooks[s]

                dists = np.linalg.norm(q[:, None, :] - cb[None, :, :], axis=2)
                batch_codes[:, s] = np.argmin(dists, axis=1)

            for leaf_id, idx_list in leaf_index.items():
                mmap = leaf_codes_memmap[leaf_id]

                for pos, global_id in enumerate(idx_list):
                    if batch_start <= global_id < batch_end:
                        local = global_id - batch_start
                        mmap[pos] = batch_codes[local]

        for mmap in leaf_codes_memmap.values():
            mmap.flush()
            del mmap

        gc.collect()


    def _build_index(self):
        os.makedirs(self.index_path, exist_ok=True)
        vectors = self.get_all_rows()

        num_records = self._get_num_records()
        indices = np.arange(num_records)

        max_depth = int(np.log2(self._determine_n_clusters(num_records)))

        tree_root = self._build_tree(
            vectors = vectors,
            indices = indices,
            depth = 0,
            max_depth = max_depth
        )

        leaves = []
        self._collect_leaves(tree_root, leaves)


        leaf_index = {i: leaf.indices for i, leaf in enumerate(leaves)}
        pq_codebooks = self._build_pq_codebooks(vectors)
        self._compute_compressed_codes(vectors, pq_codebooks, leaf_index)
        
        self._save_tree_structure(tree_root, leaves)

        for i, cb in enumerate(pq_codebooks):
            cb_path = os.path.join(self.index_path, f"pq_cb_{i}.dat")
            mmap_cb = np.memmap(cb_path, dtype=np.float16, mode='w+', shape=cb.shape)
            mmap_cb[:] = cb[:]
            mmap_cb.flush()
            del mmap_cb

        max_uint16 = np.iinfo(np.uint16).max

        for leaf_id, leaf_indices in leaf_index.items():
            leaf_indices = np.array(leaf_indices, dtype=np.uint32)
            leaf_indices.sort()

            diffs = np.diff(leaf_indices, prepend=0)

            if diffs.max() <= max_uint16:
                deltas = diffs.astype(np.uint16)
            else:
                deltas = diffs.astype(np.uint32)

            path = os.path.join(self.index_path, f"deltas_{leaf_id}.dat")
            with open(path, 'wb') as f:
                np.savez_compressed(f, deltas=deltas)

        del vectors, pq_codebooks
        gc.collect()


    def _load_index(self):
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

        return pq_codebooks

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
                return data.cumsum().astype(np.int32)
        return None

    def _compute_pq_distances(self, query, pq_codebooks):
        query = query.astype(np.float32)
        distance_tables = []

        for i in range(self.n_subvectors):
            start = i * self.subvector_dim
            end = (i + 1) * self.subvector_dim
            q = query[start:end]
            cb = pq_codebooks[i].astype(np.float32)
            numer = cb @ q          
            sim = numer / (np.linalg.norm(q) + 1e-10)
            distance_tables.append((1.0 - sim).astype(np.float32))
        return distance_tables
    
    def _collect_leaves(self, node, leaf_list):
        if node.is_leaf():
            leaf_list.append(node)
        else:
            self._collect_leaves(node.left, leaf_list)
            self._collect_leaves(node.right, leaf_list)
    
    def _save_tree_structure(self, tree_root, leaves):
        import pickle
        tree_path = os.path.join(self.index_path, "tree_structure.pkl")
        with open(tree_path, 'wb') as f:
            pickle.dump((tree_root, leaves), f)
    
    def _load_tree_structure(self):
        import pickle
        tree_path = os.path.join(self.index_path, "tree_structure.pkl")
        if os.path.exists(tree_path):
            with open(tree_path, 'rb') as f:
                tree_root, leaves = pickle.load(f)
                return tree_root, leaves
        else:
            raise FileNotFoundError(f"Tree structure not found at {tree_path}")

    def _leaf_center(self, leaf: TreeNode, vectors: np.ndarray) -> np.ndarray:
        return np.mean(vectors[leaf.indices], axis=0)
    
    def _compute_leaf_heap(self, leaf_id, leaf_indices, distance_tables, top_heap, n_take, max_batch_size=5000):
        codes_path = os.path.join(self.index_path, f"index_codes_{leaf_id}.dat")
        if not os.path.exists(codes_path):
            return

        codes_mmap = np.memmap(codes_path, dtype=np.uint8, mode='r', shape=(len(leaf_indices), self.n_subvectors))
        total_codes = len(codes_mmap)
        batch_size = min(max_batch_size, total_codes)

        for batch_start in range(0, total_codes, batch_size):
            batch_end = min(batch_start + batch_size, total_codes)
            batch_codes = codes_mmap[batch_start:batch_end]
            batch_indices = leaf_indices[batch_start:batch_end]

            dist = np.zeros(len(batch_codes), dtype=np.float32)
            for i in range(self.n_subvectors):
                dist += distance_tables[i][batch_codes[:, i]]

            m = min(n_take, len(batch_codes))
            top_local = np.argpartition(dist, m-1)[:m] if m < len(batch_codes) else np.arange(len(batch_codes))
            for d_val, idx_val in zip(dist[top_local], batch_indices[top_local]):
                if len(top_heap) < n_take:
                    heapq.heappush(top_heap, (-d_val, int(idx_val)))
                elif d_val < -top_heap[0][0]:
                    heapq.heapreplace(top_heap, (-d_val, int(idx_val)))

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5, n_probes=None):
        query = np.array(query, dtype=np.float32).flatten()
        query /= np.linalg.norm(query) + 1e-10

        tree_root, leaves = self._load_tree_structure()
        pq_codebooks = self._load_index()
        distance_tables = self._compute_pq_distances(query, pq_codebooks)

        if n_probes is None:
            n_probes = len(leaves)
        
        selected_leaves = list(range(min(n_probes, len(leaves))))

        top_heap = []
        n_take = min(top_k * 100, 500)

        for leaf_id in selected_leaves:
            leaf_indices = leaves[leaf_id].indices
            self._compute_leaf_heap(leaf_id, leaf_indices, distance_tables, top_heap, n_take)

        del distance_tables, pq_codebooks, tree_root, leaves

        if not top_heap:
            return []

        top_heap.sort(key=lambda x: x[0])
        candidate_ids = [idx for _, idx in top_heap[:n_take]]
        del top_heap

        final_heap = []
        batch_size = 200
        
        for i in range(0, len(candidate_ids), batch_size):
            batch_ids = candidate_ids[i:i+batch_size]
            for idx in batch_ids:
                vec = self.get_one_row(idx)
                sim = np.dot(query, vec)
                
                if len(final_heap) < top_k:
                    heapq.heappush(final_heap, (sim, idx))
                elif sim > final_heap[0][0]:
                    heapq.heapreplace(final_heap, (sim, idx))

        final_results = [idx for _, idx in sorted(final_heap, reverse=True)]
        return final_results
    
    def _cal_score(self, vec1, vec2):
        return np.dot(vec1, vec2)