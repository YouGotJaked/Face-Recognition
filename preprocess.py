import sys
import numpy as np
import cv2
import glob
import re

"""
    Read images from provided path.
    Args:
        path
    Return:
        images, labels (numpy.ndarray)
"""
def read_images(path):
    print("Reading images from " + path, end="...")
    # Create array of array of images.
    images = []
    labels = []
    # Iterate through all files in the given path
    for file in glob.iglob(path, recursive=True):
        pattern = re.search('s[0-9]{1,2}', file) # search for subject dir
        labels.append(pattern.group(0)) # add subject to label list
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # load image from file as vector of length D = 112 x 92
        # type = <class 'numpy.ndarray'>
        if im is None :
            print("image:{} not read properly".format(path))
        else :
            # Add image to list
            images.append(im)

    print("DONE")

    # Exit if no image found
    if len(images) == 0 :
        print("No images found")
        sys.exit(0)
    
    print("{} files read.".format(len(images)))
    return images, labels

"""
    Stack 6 training images of all 10 subjects to form a matrix of size 10304×60
    Args:
        images
    Returns:
        matrix of size 10304x60
"""
def create_data_matrix(images):
    print("Converting images to vector format", end="...")
    num_img = len(images)
    size = images[0].shape
    # create np array of size D x num_imgs
    data_matrix = np.zeros((num_img, size[0] * size[1]), dtype=np.float32)
    for i in range(num_img):
        image = images[i].flatten()
        data_matrix[i,:] = image
    print("DONE")
    print("Resulting vector is of size {}".format(data_matrix.shape))
    return data_matrix
