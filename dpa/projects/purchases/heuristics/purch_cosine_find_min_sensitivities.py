from hashlib import sha3_256 as hashfunc
import numpy as np
import os
from os import path
from tensorflow.keras.datasets.mnist import load_data
import operator

# change this static variable to change size of D and D' (either 100 or 10000)
SIZE_D = 1000
place =0

filepath = os.getcwd()+f"/purch_cosines_dist/partial_dicts"
min_per_picture = {}

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

list_of_dicts = []
for xi, h in enumerate(X_hashed_hex[:SIZE_D]):
    key=h
    list_of_dicts.append(np.load(path.join(filepath, f"{key}.npz"), allow_pickle=True)["lookup"].item(0))

# set the min value in list_of_dicts to zero so that we can choose the minimum to find the largest place
for i in range(place):
    list_of_minimums = []
    list_of_argmin = []
    for dicti in list_of_dicts:
        mylist = list(dicti.values())[SIZE_D:]
        #print(mylist)
        list_of_argmin.append(np.argmin(mylist))
        #dicti_list = list(dicti.values())
        #print(dicti_list)
        list_of_minimums.append(np.min(mylist))
    arg_img = np.argmin(list(list_of_minimums))
    print(list_of_minimums)
    print(arg_img)
    print(list_of_minimums[arg_img])
    temp_key = list(list_of_dicts[arg_img].keys())[SIZE_D+list_of_argmin[arg_img]]
    list_of_dicts[arg_img][temp_key] = 200#np.inf

#for each picture in D find min distance to each other picture in Dataset w/o D
for xi, h in enumerate(X_hashed_hex[:SIZE_D]):
    key=h
    pic_dists = list_of_dicts[xi]#np.load(path.join(filepath, f"{key}.npz"), allow_pickle=True)["lookup"].item(0)
    argmin_xi = np.argmin(list(pic_dists.values())[SIZE_D:])
    min_dist = min(list(pic_dists.values())[SIZE_D:])
    print(min_dist)
    print(list(pic_dists.keys())[SIZE_D+argmin_xi])
    min_dist_image = list(pic_dists.keys())[SIZE_D+argmin_xi]

    key=X_hashed[xi]
    min_per_picture[key] = [min_dist_image, min_dist]
save_filepath = f"./"
np.savez_compressed(path.join(save_filepath, f"min_per_picture_cosine_set_size_{SIZE_D}.npz"), lookup=min_per_picture)

#find hash and distance for pair with largest distance
# array of hash of picture b and distance
largest_diff = min(list(min_per_picture.values()), key=lambda x:x[1])
print(np.array(list(min_per_picture.values()))[:,1])
key_largest_diff_index = np.argmin(np.array(list(min_per_picture.values()))[:,1])

# key for largest difference (picture a of comparison)
key_largest_diff =list(min_per_picture.keys())[key_largest_diff_index]
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

save_filepath = os.getcwd()+f"/"
np.savez_compressed(path.join(save_filepath, f"original_data_cosine_set_size_{SIZE_D}_last_place_{place+1}.npz"), lookup=X[:SIZE_D])
np.savez_compressed(path.join(save_filepath, f"alt_data_cosine_set_size_{SIZE_D}_last_place_{place+1}.npz"), lookup=alt_set)