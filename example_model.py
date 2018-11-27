import os
from model import TrainingModel, TestingModel

DIR = os.getcwd()
TESTING_DIR = DIR + '/testing/**/*.pgm'
TRAINING_DIR = DIR + '/training/**/*.pgm'
K = [1, 2, 3, 6, 10, 20, 30]

def main():
    training = TrainingModel(TRAINING_DIR, K)
    testing = TestingModel(TESTING_DIR, K, training.knn)
    testing.predict

if __name__ == '__main__':
    main()
