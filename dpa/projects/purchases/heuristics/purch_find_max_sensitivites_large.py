"""
    For every picture in orig dataset identify an alternate picture with maximum euclidean distance
"""
from hashlib import sha3_256 as hashfunc
import numpy as np
import os
from os import path
from tensorflow.keras.datasets.mnist import load_data
import operator

# change this static variable to change size of D and D' (either 100 or 10000)
SIZE_D = 1000

filepath = f"./purch_cosines_dist"
max_per_picture = {}

# hash function wrapper
def hash_val(val, hex=True):
    m = hashfunc()
    m.update(val.data)
    if hex:
        key = m.hexdigest()
    else:
        key = m.digest()
    return key

#load purchases data and consider first 80k as training pool
X = np.load('./shokri_purchases_100_classes.npz')
X = X['x'][:80000]

X_hashed = [hash_val(xi, False) for xi in X]
X_hashed_hex = [hash_val(xi, True) for xi in X]

print(np.shape(X_hashed_hex))
#for each picture in D find max distance to each other picture in Dataset w/o D
for xi, h in enumerate(X_hashed_hex[:SIZE_D]):
    key=h
    pic_dists = np.load(path.join(filepath, f"{key}.npz"), allow_pickle=True)["lookup"].item(0)
    argmax_xi = np.argmax(list(pic_dists.values())[SIZE_D:])
    max_dist = max(list(pic_dists.values())[SIZE_D:])
    print(max_dist)
    print(list(pic_dists.keys())[SIZE_D+argmax_xi])
    max_dist_image = list(pic_dists.keys())[SIZE_D+argmax_xi]

    key=X_hashed[xi]
    max_per_picture[key] = [max_dist_image, max_dist]
save_filepath = f"./"
np.savez_compressed(path.join(save_filepath, f"max_per_picture_cosine.npz"), lookup=max_per_picture)

#find hash and distance for pair with largest distance
# array of hash of picture b and distance
largest_diff = max(list(max_per_picture.values()), key=lambda x:x[1])
print(np.array(list(max_per_picture.values()))[:,1])
key_largest_diff_index = np.argmax(np.array(list(max_per_picture.values()))[:,1])

# key for largest difference (picture a of comparison)
key_largest_diff =list(max_per_picture.keys())[key_largest_diff_index]
print(key_largest_diff)
print(largest_diff[0])
print(largest_diff[1])

d_index = -1
non_d_index = -1

#find the picture with those hashes
for i, h in enumerate(X_hashed):
    if h==key_largest_diff:
        d_index = i
    if h==largest_diff[0]:
        non_d_index = i
    if d_index != -1 and non_d_index != -1:
        break

#replace the picture in the fixed original data set with the picture resulting in largest sensitivity
alt_set = np.copy(X[:SIZE_D])
alt_set[d_index] = X[non_d_index]

print(d_index)
print(non_d_index)

save_filepath = f"./"
np.savez_compressed(path.join(save_filepath, f"original_data_set_cosine_size_{SIZE_D}_place_1.npz"), lookup=X[:SIZE_D])
np.savez_compressed(path.join(save_filepath, f"alt_data_set_cosine_size_{SIZE_D}_place_1.npz"), lookup=alt_set)