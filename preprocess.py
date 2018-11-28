"""preprocess.py - module to process input data before applying algorithm."""

import sys
import glob
import re
import numpy as np
import cv2

def read_images(path):
    """Read images from provided path.
    
    Iterate through all files in the given path.
    Perform a regex search for the subject directory.
    Add subject to list of subject labels.
    Read image from file as grayscale.
    Add image to list if read correctly.
    
    Args:
        path (str): directory of image files
    Return:
        the numpy array of images, the list of their respective labels
    """
    print("------------ PREPROCESSING ------------")
    print("Reading images from " + path, end="...")

    images = []
    labels = []
    
    for file in glob.iglob(path, recursive=True):
        subject = search_subject(file)
        labels.append(subject)
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if im is None :
            print("Image: {} not read properly".format(file))
        else :
            images.append(im)

    # Exit if no images found
    if images is None:
        print("No images found")
        sys.exit(0)

    print("DONE")
    print("{} files read.".format(len(images)))
    return images, labels

def create_data_matrix(images):
    """Stack N training images of all 10 subjects to form a matrix of size D X N.
    
    Create an all-zero array to allocate space for images.
    Flatten image to 1D vector.
    Add flattened image to matrix.
    
    Args:
        images (numpy.ndarray): array of image files
    Returns:
        matrix of flattened images with size D x N
    Notes:
        N is the number of images in `images`.
        For the training data, N = 60.
        For the testing data, N = 40.
    """
    print("Converting images to vector format", end="...")
    size = images[0].shape
    D = size[0] * size[1] # D = 112 x 92 = 10304
    data_matrix = np.zeros((len(images), D), dtype=np.float32)
    
    for i,_ in enumerate(images):
        print(images[i].shape)
        image = images[i].flatten()
        print(image.shape)
        data_matrix[i,:] = image
    
    print("DONE")
    print("Resulting vector is of size {}".format(data_matrix.shape))
    return data_matrix

def search_subject(file):
    """ Perform a regex search for the current subject.
        
    Args:
        file (str): full path to file
    Returns:
        the subject of the current file
    """
    subject_directory = re.search('s[0-9]{1,2}', file)
    return subject_directory.group(0)
