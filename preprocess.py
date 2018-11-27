import sys
import numpy as np
import cv2
import glob
import re

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
        images, labels (numpy.ndarray)
    """
    print("------------ PREPROCESSING ------------")
    print("Reading images from " + path, end="...")

    images = []
    labels = []
    
    for file in glob.iglob(path, recursive=True):
        subject = search_subject(file)
        labels.append(subject) # add subject to label list
        
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # load image from file as vector of length D = 112 x 92
        if im is None :
            print("image:{} not read properly".format(path))
        else :
            # Add image to list
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
    
    Create an all-zero array of size
    
    Args:
        images (numpy.ndarray): array of image files
    Returns:
        numpy.ndarray: matrix of size 10304x60
    """
    print("Converting images to vector format", end="...")
    size = images[0].shape
    D = size[0] * size[1]
    data_matrix = np.zeros((len(images), D), dtype=np.float32)
    
    for i,_ in enumerate(images):
        image = images[i].flatten()
        data_matrix[i,:] = image
    
    print("DONE")
    print("Resulting vector is of size {}".format(data_matrix.shape))
    return data_matrix

def search_subject(file):
# regex search for subject dir
    subject_directory = re.search('s[0-9]{1,2}', file) # regex search for subject dir
    return subject_directory.group(0)
