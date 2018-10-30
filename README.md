# Face-Recognition
You are given a face image database of 10 subjects. Each subject has 10 images of size 112×92. 
Use principal-component analysis (PCA) for dimensionality reduction. 
Find the subspaces of rank K=1,2,3,6,10,20, and 30. 
Project the face images to the subspace and apply the nearest-neighbor classifier in the rank-K subspace.
For each subject, use face images 1,3,4,5,7,9 as the training images, and face images 2,6,8,10 as the testing images. 

Convert each image to a vector of length D=112×92=10304.
Stack 6 training images of all 10 subjects to form a matrix of size 10304×60.
Apply PCA to this data matrix with different rank values.
