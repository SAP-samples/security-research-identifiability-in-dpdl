from hashlib import sha3_256 as hashfunc
import numpy as np
import os
from os import path
from tensorflow.keras.datasets.mnist import load_data
import operator

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
    sum_dists = sum(list(pic_dists.values())[:SIZE_D])
    print(sum_dists)

    key=X_hashed[xi]
    max_per_picture[key] = sum_dists


#find hash and distance for pair with largest distance
# array of hash of picture b and distance
largest_diff = max(list(max_per_picture.values()))
print(np.array(list(max_per_picture.values())))
key_largest_diff_index = np.argmax(np.array(list(max_per_picture.values())))

# key for largest difference (picture a of comparison)
key_largest_diff =list(max_per_picture.keys())[key_largest_diff_index]
print(key_largest_diff)
print(largest_diff)

d_index = -1

#find the picture with those hashes
for i, h in enumerate(X_hashed):
    if h==key_largest_diff:
        d_index = i
    if d_index != -1:
        break

#replace the picture in the fixed original data set with the picture resulting in largest sensitivity
alt_set = np.copy(X[:SIZE_D])
alt_set = np.delete(alt_set, d_index,0)

print(d_index)

print(alt_set.shape)

save_filepath = os.getcwd()+f"/"
np.savez_compressed(path.join(save_filepath, f"unbounded_original_data_set_ssim_size_{SIZE_D}.npz"), lookup=X[:SIZE_D])
np.savez_compressed(path.join(save_filepath, f"unbounded_alt_data_set_ssim_size_{SIZE_D}.npz"), lookup=alt_set)