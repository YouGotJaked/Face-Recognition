import cv2
import numpy as np

# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.eig.html

# apply PCA to this matrix with different rank values for dimensionality reduction

# find subspaces of rank K = 1, 2, 3, 6, 10, 20, and 30

# project face images to the subspace and apply the nearest-neighbor classifier in the rank-K subspace

# use 1, 3, 4, 5, 7, 9 for training

# use 2, 6, 8, 10 for testing
