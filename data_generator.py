import numpy as np

def get_all_rows(db_path) -> np.ndarray:
    vectors = np.memmap(db_path, dtype=np.float32, mode='r', shape=(20*(10**6), 64))
    return np.array(vectors)

vectors = get_all_rows('OpenSubtitles_en_20M_emb_64.dat')

random_idx = np.random.randint(0, vectors.shape[0], size=10**7)
# sample
sampled_vectors = vectors[random_idx]
# shuffle
np.random.shuffle(sampled_vectors)
# add small noise
sampled_vectors += np.random.normal(0, 0.01, sampled_vectors.shape)
# normalize
sampled_vectors /= np.linalg.norm(sampled_vectors, axis=1, keepdims=True) + 1e-10
# save
with open('OpenSubtitles_en_10M_emb_64.dat', 'wb') as f:
    sampled_vectors.tofile(f)