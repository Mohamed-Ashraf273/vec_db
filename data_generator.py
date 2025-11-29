import numpy as np

def generate_15M_from_20M(src_path, dst_path):
    src_vectors = np.memmap(src_path, dtype=np.float32, mode='r', shape=(20*10**6, 64))
    
    np.random.seed(42)
    random_idx = np.random.randint(0, 20*10**6, size=15*10**6)
    
    dst_vectors = np.memmap(dst_path, dtype=np.float32, mode='w+', shape=(15*10**6, 64))
    
    batch_size = 100000
    for i in range(0, 15*10**6, batch_size):
        end_idx = min(i + batch_size, 15*10**6)
        batch_indices = random_idx[i:end_idx]
        
        batch = np.array(src_vectors[batch_indices])
        batch += np.random.normal(0, 0.01, batch.shape)
        
        batch /= np.linalg.norm(batch, axis=1, keepdims=True) + 1e-10
        
        dst_vectors[i:end_idx] = batch
        
        if (i // batch_size) % 10 == 0:
            print(f"Progress: {i}/{15*10**6}")
    
    dst_vectors.flush()
    print("Done!")

if __name__ == "__main__":
    generate_15M_from_20M('saved_db_20M.dat', 'saved_db_15M.dat')