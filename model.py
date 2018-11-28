"""model.py - module to store the training and testing models."""

import numpy as np
from preprocess import read_images, create_data_matrix
from subspace import pca, knn, predict, accuracy

class Model():
    """Base class to define data model."""
    def __init__(self, path, K):
        """Preprocess images in the data model."""
        self.path = path
        self.images, self.labels = read_images(self.path)
        self.matrix = create_data_matrix(self.images)
        self.K = K

class TrainingModel(Model):
    """Child class to define training data model."""
    def __init__(self, path, K):
        """Inherit constructor from base class."""
        super(TrainingModel, self).__init__(path, K)

class TestingModel(Model):
    """Child class to define testing data model."""
    def __init__(self, path, K):
        super(TestingModel, self).__init__(path, K)
    
    def reduce(self, training_matrix):
        """Perform PCA and dimensional reduction for all ranks in K."""
        return [pca(training_matrix, self.matrix, rank) for rank in self.K] # calculate pca for each rank
    
    def knn(self, reduced, labels):
        """Returns the KNN classifier for all ranks in K."""
        return [knn(reduced[i][0], labels) for i,_ in enumerate(self.K)] # calculate knn for each rank

    def predict(self, knn, reduced):
        """Returns the predicted subject label for all ranks in K."""
        return [predict(knn[i], self.K[i], reduced[i][1], self.labels) for i,_ in enumerate(self.K)]
    
    def accuracy(self, knn, reduced):
        """Returns the accuracy of each prediction for all ranks in K."""
        return [accuracy(knn[i], self.K[i], reduced[i][1], self.labels) for i,_ in enumerate(self.K)]
