# Cyberlib---ML-Bootcamp
To a Machine Learning Library from scratch

## **Objectives:**

Implement from scratch  
- Linear Regression
- Polynomial Regression
-  Logistic Regression
-  KNN
-  K-Means Clustering
-  Neural Networks
  A. Classification  
  B. Linear Regression  
  C. Polynomial Regression  

## **Implementation Details:**

### 1. Linear Regression:
- A basic linear regression algorithm to calculate the parameters (theta), using to fit the training data with least cost and be able to predict values for similar data (like testing data).
- Scaling used: Mean scaling with standard deviation
- Metric used:
  - Mean Squared Error (MSE) = 2794.58
- Input as :
  - train(X_train, y_train, num_iter, alpha)  
   X-train  --> Design matrix of features of trainig data  
   y_train  --> Target value vector of trainig data  
   num_iter --> number of iterations  
   alpha    --> learning rate  

  - predict( X_test, y_test, y_train, theta)  
   X-test   --> Design matrix of features of testing data  
   y_test   --> Target value vector of testing data  
   y_train  --> Target value vector of trainig data (for descaling)  
   theta    --> parameter obtained after training  
   
- Problems faced :
   - Introduction to Google Colab
   - Importing files into colab

- figures:
  - ![Cost vs iterations]()
  - ![Hypothesis (line plot) and Target values (scatter points) of first 10 test examples]()

### 1. Polynomial Regression:
- Similar to linear regression, this algorithm calculates parameter and uses it to predict further values but also used additional higher degree terms (here upto 3rd degree) as additional features to better fit the data. Hence, regularisation is also used in order to prevent parameter from overfitting the training data.
- Scaling used: Mean scaling with standard deviation
- Metric used:
  - Mean Squared Error (MSE) = 61.64
- Input as :
  - train(X_train, y_train, num_iter, alpha)  
   X-train   --> Design matrix of features of trainig data  
   y_train   --> Target value vector of trainig data  
   num_iter  --> number of iterations  
   alpha     --> learning rate
   reg_coeff --> regularisation coefficient

  - predict( X_test, y_test, y_train, theta)  
   X-test   --> Design matrix of features of testing data  
   y_test   --> Target value vector of testing data  
   y_train  --> Target value vector of trainig data (for descaling)  
   theta    --> parameter obtained after training  
   
- Problems faced :
   - Appending higher degree terms in design matrix

- figures:
  - ![Cost vs iterations]()
  - ![Hypothesis (line plot) and Target values (scatter points) of first 10 test examples]()

### 3. Logistic Regression:
- Classification algorithm, using sigmoid as activation function to calculate probabilites of a data point (from EMNIST dataset) to belong in any of 26 classes (Alphabets)
- Scaling used: Mean scaling with standard deviation (if standard deviation for any feature, std = 0, then std is taken to be, std = 1)
- Metric used:
  - Log Loss = 0.67
  - Accuracy = 57.79 %
- Input as :
  - train(X_train, y_train, num_iter, alpha)  
   X-train   --> Design matrix of features of trainig data  
   y_train   --> Target value vector of trainig data  
   num_iter  --> number of iterations  
   alpha     --> learning rate  
   reg_coeff --> regularisation coefficient

  - predict(X, y, theta, num_class):  
  X           --> Design matrix of features of testing data  
  y           --> Target value vector of testing data  
  theta       --> parameter obtained after training  
  num_classes --> number of classes   
   
- Problems faced :
   - Converting class vector to probability matrix and vice versa

- figures:
  - ![Cost vs iterations]()

### 4. K Nearest Neighbours (KNN):
- Classifying data comparing data points from testing data to each of training data, finding K nearest neighbours (using eiclidian distance) and assigning datapoint to the class of the neighbour with highest mode (frequency)
- Metric used:
   - Accuracy = 
- Input as :
  - test(X_test, y_test, X_train, y_train, num_classes, k=5):  
  X_test --> Design matrix of test data  
  y_test --> target class vector  
  X_train --> Design matrix of training (reference) data  
  y_train --> reference class vector  
  K --> Number of neighbours  
   
- Problems faced :
   - Implementing algorithm
   - Long testing time

- figures:
  - ![Cost vs iterations]()
  - ![Hypothesis (line plot) and Target values (scatter points) of first 10 test examples]()

### 5. K - Means Clustering:
- Clustering algorithm to group similar data points together, by calculating euclidian distance from centroid and updating centroid in each iteration.
- Metric used:
- Dunn Index = 0.2244915204008792
- Input as :
  - Kmeans (X, num_iter, k = 5):  
  X        --> mxn design matrix of data points with n features and m points  
  num_iter --> number of iterations  
  k        --> number of clusters   
   
- Problems faced :
   - Assigning cluster to a data point i.e., segregating data points into clusters

- figures:
  - ![Cost vs iterations]()
  - ![Hypothesis (line plot) and Target values (scatter points) of first 10 test examples]()

### 6. Neural Network:
- Implemented a 2 layer neural network (1-hidden and 1-output) to perform classification, linear regression, polynomial regression

      
    
