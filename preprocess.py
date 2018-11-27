import os
import sys
import cv2
import numpy as np
import glob
import re
import sklearn
from sklearn import neighbors
from sklearn.decomposition import PCA

# Constants
DIR = os.getcwd()
TEST_DIR = DIR + '/testing/**/*.pgm'
TRAIN_DIR = DIR + '/training/**/*.pgm'
K = [1, 2, 3, 6, 10, 20, 30]
SUBJECTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]

##############
# PREPROCESS #
##############


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
"""
def create_data_matrix(images):
    print("Converting images to vector format", end="...")
    num_img = len(images)
    size = images[0].shape
    # create np array of size D x num_img
    data_matrix = np.zeros((num_img, size[0] * size[1]), dtype=np.float32)
    for i in range(0, num_img):
        image = images[i].flatten()
        data_matrix[i,:] = image
    print("DONE")
    print("Resulting vector is of size {}".format(data_matrix.shape))
    return data_matrix

"""
    Apply Principal Component Analysis to data matrix with rank `rank`
"""
def pca(data_matrix, rank):
    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA with rank {}".format(rank), end="...")
    mean, eigenvectors = cv2.PCACompute(data_matrix, mean=None, maxComponents=rank)
    print ("DONE")
    return mean, eigenvectors

"""
    Project face image into subspace
    Args:
        data
        mean
        eigenvectors
    Returns:
        numpy.ndarray:
"""
def project(data, mean, eigenvectors):
    return cv2.PCAProject(data, mean, eigenvectors)

"""
    Apply KNN in rank-K subspace
"""
def euclideanDistance(a, b, length):
    distance = 0
    for i in range(length):
        distance += pow((a[i] - b[i]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def knn(p, q):
    return None

"""
    Predict
    Args:
    Returns:
"""
def predict():
    return None

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

train_images, train_labels = read_images(TRAIN_DIR)
test_images, test_labels = read_images(TEST_DIR)
#print(images[0])
#print(labels)
#sz = images[0].shape
train_images = create_data_matrix(train_images)
test_images = create_data_matrix(test_images)


np.save("face_train_images", train_images)
np.save("face_train_lbls", train_labels)
np.save("face_test_images", test_images)
np.save("face_test_lbls", test_labels)

for rank in K:
    pca = PCA(n_components=rank)
    pca.fit(train_images)
    pca.fit(test_images)
    train_reduced = pca.transform(train_images)
    test_reduced = pca.transform(test_images)
    knn = neighbors.KNeighborsClassifier()
    k_fit = knn.fit(train_reduced, train_labels)
    print("Rank: {}, Accuracy: {} ".format(rank, k_fit.score(test_reduced, test_labels)))

###########
# TESTING #
###########
