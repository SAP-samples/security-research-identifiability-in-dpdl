"""
    For every picture in orig dataset identify an alternate picture with maximum euclidean distance
"""
from hashlib import sha3_256 as hashfunc
import numpy as np
import os
from os import path
from tensorflow.keras.datasets.mnist import load_data
import statistics
import operator
import struct

# change this static variable to change size of D and D' (either 100 or 10000)
SIZE_D = 100

filepath = os.getcwd()+f"/mnist_ssim_dist/partial_dicts"
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

# load mnist data
(x_train, _), (x_test, _) = load_data()
X = np.concatenate((x_train, x_test))
X = X.reshape((len(X), 28 * 28))
#test data
#num_fake_data = 10
#X = np.tile(np.arange(num_fake_data).reshape(num_fake_data, 1), (1, num_fake_data))

X_hashed = [hash_val(xi, False) for xi in X]
X_hashed_hex = [hash_val(xi, True) for xi in X]

print(np.shape(X_hashed_hex))
#for each picture in D find max distance to each other picture in Dataset w/o D
list_of_dicts = []
for xi, h in enumerate(X_hashed_hex[:SIZE_D]):
    key=h
    list_of_dicts.append(np.load(path.join(filepath, f"{key}.npz"), allow_pickle=True)["lookup"].item(0))


for xi, h in enumerate(X_hashed_hex[:SIZE_D]):
    key=h
    pic_dists = list_of_dicts[xi]#np.load(path.join(filepath, f"{key}.npz"), allow_pickle=True)["lookup"].item(0)
    curr_list = list(pic_dists.values())[SIZE_D:]
    argmax_xi = curr_list.index(np.percentile(curr_list,50,interpolation='nearest'))#np.argmax(list(pic_dists.values())[SIZE_D:])
    max_dist = statistics.median(list(pic_dists.values())[SIZE_D:])# max(list(pic_dists.values())[SIZE_D:])
    print(max_dist)
    print(list(pic_dists.keys())[SIZE_D+argmax_xi])
    max_dist_image = list(pic_dists.keys())[SIZE_D+argmax_xi]

    key=X_hashed[xi]
    max_per_picture[key] = [max_dist_image, max_dist]
save_filepath = f"./"
np.savez_compressed(path.join(save_filepath, f"max_per_picture_ssim_set_size_{SIZE_D}.npz"), lookup=max_per_picture)

#find hash and distance for pair with largest distance
# array of hash of picture b and distance
print("HERE", np.array(list(max_per_picture.values()))[:,1].astype(np.float))
largest_diff = np.median(np.array(list(max_per_picture.values()))[:,1].astype(np.float))#max(list(max_per_picture.values()), key=lambda x:x[1])
arr_maxes = list(max_per_picture.values())
list_now = list(np.array(list(max_per_picture.values()))[:,1])
key_first = arr_maxes[list_now.index(np.percentile(arr_maxes,50,interpolation='nearest'))][0]


key_largest_diff_index = list_now.index(np.percentile(list_now,50,interpolation='nearest'))#np.argmax(np.array(list(max_per_picture.values()))[:,1])

# key for largest difference (picture a of comparison)
key_largest_diff =list(max_per_picture.keys())[key_largest_diff_index]
print(key_largest_diff)
print(largest_diff)
print(key_first)

d_index = -1
non_d_index = -1

#find the picture with those hashes
for i, h in enumerate(X_hashed):
    if h==key_largest_diff:
        d_index = i
    if h==key_first:
        non_d_index = i
    if d_index != -1 and non_d_index != -1:
        break

#replace the picture in the fixed original data set with the picture resulting in largest sensitivity
alt_set = np.copy(X[:SIZE_D])
alt_set[d_index] = X[non_d_index]

print(d_index)
print(non_d_index)

save_filepath = os.getcwd()+f"/"
np.savez_compressed(path.join(save_filepath, f"original_data_set_ssim_size_{SIZE_D}_median.npz"), lookup=X[:SIZE_D])
np.savez_compressed(path.join(save_filepath, f"alt_data_set_ssim_size_{SIZE_D}_median.npz"), lookup=alt_set)