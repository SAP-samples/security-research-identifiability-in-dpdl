"""
    Loads keras mnist dataset, calculates all between-picture distances and saves one dictionary per picture with the distances to all other pictures.
    To get the dictionary for a specific picture, hash the data of the flattened picture with
    
    ```
    m = hashlib.sha3_256()
    m.update(img.data)
    key = m.hexdigest()
    ```
    and load the corresponding file with
    ```
    lookup = np.load(path.join(filepath, f"{key}.npz"), allow_pickle=True)["lookup"].item(0)
    ```
    To look up a specific flattened picture in the loaded dictionary use the binary digest as key, i.e.,
    ```
    m = hashlib.sha3_256()
    m.update(img.data)
    key = m.digest()
    lookup[key]
    ```
    Don't reuse m!
"""
#%%
from datetime import datetime
from hashlib import sha3_256 as hashfunc
from os import getcwd, makedirs, path, sched_getaffinity

import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tqdm.contrib.concurrent import process_map

global X, batch_size, len_X, filepath

#%%
# use size_subsample for tests.
# If bigger then number of records, works on the whole dataset.
size_subsample = 100000

(x_train, _), (x_test, _) = load_data()
X = np.concatenate((x_train, x_test))
# flatten data
X = X.reshape((len(X), 28 * 28))[:size_subsample]

#%%
# num_fake_data = 10
# X = np.tile(np.arange(num_fake_data).reshape(num_fake_data, 1), (1, num_fake_data))

#%%
def mat_mul(start_idx: int) -> None:

    # repeat batch to work on, i.e., [1,1,1,2,2,2]
    X_first = np.repeat(X[start_idx : start_idx + batch_size], len_X, axis=0)
    # tile all images, i.e, [1,2,3,1,2,3]
    X_second = np.tile(X, (batch_size, 1))
    # distance calculation, change if needed
    X_dist = np.sqrt(np.sum(np.square(X_first - X_second), axis=1))
    X_dist = X_dist.astype(np.float16)

    # loop over all pictures in the batch
    # lookup range helpful, when the last batch is not full
    lookup_range = batch_size if start_idx + batch_size < len_X else len_X - start_idx

    X_first.flags.writeable = False
    X_second.flags.writeable = False

    for idx in range(lookup_range):
        offset = idx * len_X
        m = hashfunc()
        m.update(X_first[offset].data)
        main_hash = m.hexdigest()

        tmp = {}
        # loop over all pictures
        for jdx in range(len_X):
            m = hashfunc()
            m.update(X_second[offset + jdx].data)
            tmp_hash = m.digest()
            tmp[tmp_hash] = X_dist[offset + jdx]

        np.savez_compressed(path.join(filepath, f"{main_hash}.npz"), lookup=tmp)

    del X_first, X_second, X_dist, tmp, main_hash, tmp_hash


#%%
batch_size = 10
num_processes = len(sched_getaffinity(0))
len_X = len(X)
all_idx = np.arange(0, len_X, batch_size)

filepath = path.join(getcwd(), "partial_dicts")
makedirs(filepath, exist_ok=True)

start = datetime.now()

# the same as using multiprocessing, but shows progress bar
# with multiprocessing.Pool(num_processes) as p:
#     p.map(mat_mul, all_idx)
process_map(mat_mul, all_idx, max_workers=num_processes, chunksize=1)

stop = datetime.now()

print(f"Overall calculation took {stop - start}", flush=True)


#%%
# lookup = np.load(
#     path.join(
#         filepath,
#         "d609e40c13512f409e8075f9da4a72aef6822ad14815b601eac3fc339db2a00a.npz",
#     ),
#     allow_pickle=True,
# )["lookup"].item(0)

#%%