"""subspace.py - module to perform linear algebra operations on a data matrix."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def pca(training_matrix, testing_matrix, rank):
    """Apply Principal Component Analysis to data matrix with `rank` components
        
    Perform linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
    Fit the PCA model with the training and testing matrices.
    Apply dimensional reduction to both matrices.
    
    Args:
        training_matrix (numpy.ndarray): matrix of training data
        testing_matrix (numpy.ndarray): matrix of testing data
        rank (int): current rank
    Returns:
        the reduced data matrices
    """
    print("Calculating PCA with rank {}".format(rank), end="...")
    pca = PCA(n_components=rank)
    pca.fit(training_matrix)
    pca.fit(testing_matrix)
    print("DONE")
    return pca.transform(training_matrix), pca.transform(testing_matrix)

def knn(model, labels):
    """Apply The K-Nearest-Neighbors algorithm in rank-K subspace.
        
    Fit the model using `model` as training data and `labels` as target values.
    
    Args:
        model (numpy.ndarray): training data
        labels (list): target values
    Returns:
        the KNeighborsClassifier after fitting the training data
    """
    knn = KNeighborsClassifier(n_neighbors=11)
    return knn.fit(model, labels)

def predict(knn, rank, model, labels):
    """Predict the class labels for the provided data.
        
    Iterate through all data models.
    Display the predicted and actual subjects of each model.
    Write the results to a text file.
    
    Args:
        knn (sklearn.neighbors.classification.KNeighborsClassifier): fitted KNeighborsClassifier
        rank (int): value of the current rank
        model (numpy.ndarray): data to test against labels
        labels (list): list of labels to predict against
    """
    print("------------- RANK = {} ---------------".format(rank))
    filename = "rank" + str(rank) + ".txt"
    fileout = ""
    predicted = knn.predict(model)
    for i,_ in enumerate(predicted):
        matched = "\tMATCHED" if predicted[i] == labels[i] else ""
        print("Predicted: {}\tActual: {}{}".format(predicted[i], labels[i], matched))
        fileout += ("Predicted: " + predicted[i] + "\tActual: " + labels[i] + matched + "\n")
    
    with open(filename, "w") as text_file:
        text_file.write(fileout)

def accuracy(knn, rank, model, labels):
    """ Display the prediction accuracy for the current rank.
        
    Args:
        knn (sklearn.neighbors.classification.KNeighborsClassifier): fitted KNeighborsClassifier
        rank (int): value of the current rank
        model (numpy.ndarray): data to score against labels
        labels (list): list of labels to score against
    """
    if rank == 1:
        print("----------------- ACCURACY ------------------")
    accuracy = knn.score(model, labels)
    percentage = accuracy * 100
    print("Rank: {:<4s}\tAccuracy: {}\t{}% matched\t".format(str(rank), accuracy, round(percentage,3)))
