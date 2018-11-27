"""subspace.py - module to perform linear algebra operations on a data matrix."""

from sklearn import neighbors
from sklearn.decomposition import PCA

def pca(data_matrix, rank):
    """Apply Principal Component Analysis to data matrix with rank `rank`
    
    Args:
        data_matrix (numpy.ndarray)
        rank (int)
    Returns:
        numpy.ndarray:
    """
    print("Calculating PCA with rank {}".format(rank), end="...")
    pca = PCA(n_components=rank)
    pca.fit(data_matrix) #Fit the model with X
    reduced = pca.transform(data_matrix) # apply dimensional reduction
    print("DONE")
    return reduced

def knn(model, labels):
    """Apply KNN in rank-K subspace
    
    Args:
        model
        labels
    Returns:
        sklearn.neighbors.classification.KNeighborsClassifier
    """
    knn = neighbors.KNeighborsClassifier()
    return knn.fit(model, labels) # Fit the model using X as training data and y as target values

def predict(knn, rank, model, labels):
    """Predict the class labels for the provided data
    
    Args:
        knn
        model
    Returns:
        numpy.ndarray
    """
    print("------------- RANK = {} ---------------".format(rank))
    predicted = knn.predict(model)
    for i,_ in enumerate(predicted):
        matched = "\tMATCHED" if predicted[i] == labels[i] else ""
        print("Predicted: {}\tActual: {}{}".format(predicted[i], labels[i], matched))
    accuracy = knn.score(model, labels)
    percentage = accuracy * 100
    print("Accuracy: {}\t{}% matched".format(accuracy, round(percentage,3)))

