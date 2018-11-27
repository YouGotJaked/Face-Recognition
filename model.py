from preprocess import read_images, create_data_matrix
from subspace import pca, knn, predict

class Model():
    def __init__(self, path, K):
        self.path = path
        self.images, self.labels = read_images(self.path)
        self.matrix = create_data_matrix(self.images)
        self.K = K
        self.reduced = [pca(self.matrix, rank) for rank in self.K] # calculate pca for each rank

class TrainingModel(Model):
    def __init__(self, path, K):
        super(TrainingModel, self).__init__(path, K)
        self.knn = [knn(self.reduced[i], self.labels) for i,_ in enumerate(K)] # calculate knn for each rank

class TestingModel(TrainingModel):
    def __init__(self, path, K, knn):
        super(TestingModel, self).__init__(path, K)
        self.predict = [predict(self.knn[i], K[i], self.reduced[i], self.labels) for i,_ in enumerate(K)]

