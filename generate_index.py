from vec_db import VecDB

size = 20*10**6
print(f"SCANN Retrieval: with {size} vectors")
db = VecDB(db_size = size, index_file_path='index_20M', database_file_path='saved_db_20M.dat')
print("Index built.")