import sys
import numpy as np
import cv2
import glob
import re

def read_images(path):
    """Read images from provided path.
    
    Args:
        path
    Return:
        images, labels (numpy.ndarray)
    """
    print("Reading images from " + path, end="...")

    images = []
    labels = []
    
    # Iterate through all files in the given path
    for file in glob.iglob(path, recursive=True):
        subject_directory = re.search('s[0-9]{1,2}', file) # search for subject dir
        subject = subject_directory.group(0)
        labels.append(subject) # add subject to label list
        
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # load image from file as vector of length D = 112 x 92
        if im is None :
            print("image:{} not read properly".format(path))
        else :
            # Add image to list
            images.append(im)

    # Exit if no images found
    if len(images) == 0:
        print("No images found")
        sys.exit(0)

    print("DONE")
    print("{} files read.".format(len(images)))
    return images, labels

def create_data_matrix(images):
    """Stack 6 training images of all 10 subjects to form a matrix of size 10304×60
    Args:
        images (numpy.ndarray): array of images
    Returns:
        numpy.ndarray: matrix of size 10304x60
    """
    print("Converting images to vector format", end="...")
    size = images[0].shape
    # create np array of size D x num_imgs
    data_matrix = np.zeros((len(images), size[0] * size[1]), dtype=np.float32)
    for i,_ in enumerate(images):
        image = images[i].flatten()
        data_matrix[i,:] = image
    print("DONE")
    print("Resulting vector is of size {}".format(data_matrix.shape))
    print(type(data_matrix))
    return data_matrix
