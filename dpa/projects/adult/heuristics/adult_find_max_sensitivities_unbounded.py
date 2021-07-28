from hashlib import sha3_256 as hashfunc
import numpy as np
import os
from os import path
import operator

# change this static variable to change size of D and D' (either 100 or 10000)
SIZE_D = 1000

filepath = os.getcwd()+f"/adult_wo_na"
max_per_adult = {}

# hash function wrapper
def hash_val(val, hex=True):
    m = hashfunc()
    m.update(val.data)
    if hex:
        key = m.hexdigest()
    else:
        key = m.digest()
    return key

# load mnist data

X = np.load('./adult_wo_na.npz')
X = X['x'][:30162]

X_hashed = [hash_val(np.ascontiguousarray(xi), False) for xi in X]
X_hashed_hex = [hash_val(np.ascontiguousarray(xi), True) for xi in X]

print(np.shape(X_hashed_hex))
#for each adult in D find max distance to each other adult in Dataset w/o D
list_of_dicts = []
for xi, h in enumerate(X_hashed_hex[:SIZE_D]):
    key=h
    adult_dists = np.load(path.join(filepath, f"{key}.npz"), allow_pickle=True)["lookup"].item(0)
    argmax_xi = np.argmax(list(adult_dists.values())[SIZE_D:])
    max_dist = max(list(adult_dists.values())[SIZE_D:])
    print(max_dist)
    print(list(adult_dists.keys())[SIZE_D+argmax_xi])
    max_dist_adult = list(adult_dists.keys())[SIZE_D+argmax_xi]

    key=X_hashed[xi]
    max_per_adult[key] = [max_dist_adult, max_dist]

#find hash and distance for pair with largest distance
# array of hash of adult b and distance
largest_diff = max(list(max_per_adult.values()), key=lambda x:x[1])
print(np.array(list(max_per_adult.values()))[:,1])
key_largest_diff_index = np.argmax(np.array(list(max_per_adult.values()))[:,1])

# key for largest difference (adult a of comparison)
key_largest_diff =list(max_per_adult.keys())[key_largest_diff_index]
print(key_largest_diff)
print(largest_diff)

d_index = -1

#find the adult with those hashes
for i, h in enumerate(X_hashed):
    if h==key_largest_diff:
        d_index = i
        print("good index ", d_index)
    if d_index != -1:
        break

print("index ", d_index)

#replace the adult in the fixed original data set with the adult resulting in largest sensitivity
alt_set = np.copy(X[:SIZE_D])
alt_set = np.delete(alt_set, d_index,0)

print(d_index)

print(alt_set.shape)

save_filepath = os.getcwd()+f"/"
np.savez_compressed(path.join(save_filepath, f"unbounded_original_adult_data_set_manh_size_{SIZE_D}.npz"), lookup=X[:SIZE_D])
np.savez_compressed(path.join(save_filepath, f"unbounded_alt_adult_data_set_manh_size_{SIZE_D}.npz"), lookup=alt_set)