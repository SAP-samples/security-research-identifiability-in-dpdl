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
from tqdm.contrib.concurrent import process_map

# needed when ssim distance is wanted
global X, batch_size, len_X, filepath

#%%
# use size_subsample for tests.
# If bigger then number of records, works on the whole dataset.
size_subsample = 1000

X = np.load('./shokri_purchases_100_classes.npz')
X = X['x']

#%
def mat_mul(start_idx: int) -> None:

    # repeat batch to work on, i.e., [1,1,1,2,2,2]
    X_first = np.repeat(X[start_idx : start_idx + batch_size], len_X, axis=0)
    # tile all images, i.e, [1,2,3,1,2,3]
    X_second = np.tile(X, (batch_size, 1))

    # distance calculation, change if needed
    # MSE
    # X_dist = np.sqrt(np.sum(np.square(X_first - X_second), axis=1))
    # X_dist = X_dist.astype(np.float16)

    # Euclidean Distance of FFT Magnitudes
    X_dist = np.sum(np.square(X_first - X_second), axis=1)
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

# change when needed
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
#         "3d4119b19137ddba3c526ac24b518377b244b2cd59639f82bcf1165e062a5c77.npz",
#     ),
#     allow_pickle=True,
# )["lookup"].item(0)

# print(lookup)
#%%
