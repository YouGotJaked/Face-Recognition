import os
import cv2
import numpy as np
import glob
from PIL import Image

DIR = os.getcwd()
IMG_DIR = DIR + '/att_faces_10/**/*.pgm'

img_lst = []
# iterate through all images
for file in glob.iglob(IMG_DIR, recursive=True):
    img_lst.append(file)

# create array of each image
img_arr = np.array( [np.array(Image.open(img).convert('L')) for img in img_lst] )

for i in img_arr:
    shape = i.shape
    flat_arr = i.ravel()
    vector = np.matrix(flat_arr)
    arr2 = np.asarray(vector).reshape(shape)
    img2 = Image.fromarray(arr2, 'L')
    img2.show()

# convert each image to a vector of length D = 112 x 92 = 10304

# stack 6 training images of all 10 subject to form a matrix of size 10304 x 60
