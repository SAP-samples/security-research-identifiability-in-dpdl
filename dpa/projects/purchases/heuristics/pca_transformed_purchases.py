from os import getcwd, makedirs, path
import numpy as np


X = np.load('../../../../data/purch/shokri_purchases_10_classes.npz')
data = X['x']

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normal_data_mean = data[:] - mean

C = np.cov(normal_data_mean.T)
eigvalues, eigvectors = np.linalg.eig(C)

pairs = [[np.abs(eigvalues[i]), eigvectors[:,i]] for i in range(len(eigvalues))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
pairs = np.reshape(pairs, (600, 2))

sum_of_variances = np.sum([i/np.sum(eigvalues) for i in eigvalues])
variance_percentage = [i/np.sum(eigvalues) for i in eigvalues]
cumulative_variance = np.cumsum(variance_percentage)

threshold = 0.9
dimensions = np.searchsorted(cumulative_variance, threshold) + 1

print("Dimensions: ", dimensions)
B = pairs[0:dimensions, 1]
B = np.vstack(B)

mean = np.reshape(mean[0:600], (600,1))
a = np.ascontiguousarray((B @ normal_data_mean.T).T)
a = np.round(a,3)

np.savez_compressed(path.join(f"../../../../data/purch", f"shokri_purchases_10_classes_pca.npz"), x=a)