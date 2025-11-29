import numpy as np
from vec_db import VecDB
import time
from dataclasses import dataclass
from typing import List
import psutil
import os

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, np_rows, top_k, num_runs):
    results = []
    ram_usages = []
    peak_ram_usages = []
    
    for _ in range(num_runs):
        query = np.random.random((1,70))
        
        process = psutil.Process(os.getpid())
        tic_mem = process.memory_info().rss  # before retrieval

        tic = time.time()
        db_ids = db.retrieve(query, top_k)
        toc = time.time()
        run_time = toc - tic

        toc_mem = process.memory_info().rss  # after retrieval

        ram_used_mb = (toc_mem - tic_mem) / (1024 ** 2)
        ram_usages.append(ram_used_mb)
        peak_ram_usages.append(ram_used_mb)  # psutil doesn't track per-call peak easily

        # actual top-k using exact cosine
        tic = time.time()
        actual_ids = np.argsort(
            np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)),
            axis=1
        ).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic
        
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    
    return results, ram_usages, peak_ram_usages


def eval(results: List[Result], ram_usages: List[float], peak_ram_usages: List[float]):
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    avg_ram_used = sum(ram_usages) / len(ram_usages)
    avg_peak_ram = sum(peak_ram_usages) / len(peak_ram_usages)

    return sum(scores) / len(scores), sum(run_time) / len(run_time), avg_ram_used, avg_peak_ram


if __name__ == "__main__":
    size = 10**3
    print(f"SCANN Retrieval: with {size} vectors")
    db = VecDB(db_size = size, index_file_path='index')

    all_db = db.get_all_rows()

    res, ram_usages, peak_ram_usages = run_queries(db, all_db, 5, 10)
    score, avg_time, avg_ram, avg_peak_ram = eval(res, ram_usages, peak_ram_usages)
    print(f"Score: {score}, Avg Time: {avg_time:.4f}s, RAM Used during retrieval: {avg_ram:.2f} MB, Peak RAM: {avg_peak_ram:.2f} MB")
