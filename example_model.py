"""example_model.py - module to demonstrate the face recognition algorithm.

This module creates a training and testing model.
It then tests the accuracy of the algorithm by comparing the testing data against the training data.
"""

import os
from model import TrainingModel, TestingModel

# CONSTANTS
DIR = os.getcwd()
TESTING_DIR = DIR + '/testing/**/*.pgm'
TRAINING_DIR = DIR + '/training/**/*.pgm'
K = [1, 2, 3, 6, 10, 20, 30]

def main():
    training = TrainingModel(TRAINING_DIR, K)
    testing = TestingModel(TESTING_DIR, K)
    reduced = testing.reduce(training.matrix)
    knn = testing.knn(reduced, training.labels)
    testing.predict(knn, reduced)
    testing.accuracy(knn, reduced)


if __name__ == '__main__':
    main()
