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
- Mean Squared Error (MSE) = 2794.586689591185
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
      
    
