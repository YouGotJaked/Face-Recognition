import numpy as np
import cv2
import sklearn
from sklearn import neighbors
from sklearn.decomposition import PCA

"""
    Apply Principal Component Analysis to data matrix with rank `rank`
    Args:
        data_matrix
        rank
    Returns:
        numpy.ndarray
"""
def pca(data_matrix, rank):
    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA with rank {}".format(rank), end="...")
    pca = PCA(n_components=rank)
    pca.fit(data_matrix) #Fit the model with X
    reduced = pca.transform(data_matrix) # apply dimensional reduction
    print("DONE")
    return reduced

"""
    Apply KNN in rank-K subspace
    Args:
        model
        labels
    Returns:
        sklearn.neighbors.classification.KNeighborsClassifier
    
"""
def knn(model, labels):
    knn = neighbors.KNeighborsClassifier()
    return knn.fit(model, labels) # Fit the model using X as training data and y as target values


"""
    Predict the class labels for the provided data
    Args:
        knn
        model
    Returns:
        numpy.ndarray
"""
def predict(knn, model, label):
    pre = knn.predict(model)
        #for i,_ in enumerate(pre):
        #print("Predicted: {}, Actual: {}".format(pre[i], label))
    return pre

"""
    Print the mean accuracy on the given test data and labels.
    Args:
        knn
        rank
        model
        labels
"""
def accuracy(knn, rank, model, labels):
    print("Rank: {}, Accuracy: {} ".format(rank, knn.score(model, labels)))
