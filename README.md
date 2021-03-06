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
- Google Colab : [https://colab.research.google.com/drive/1BvTH0D75JxnbKIE5-5i5YaxXRPVCkUcT?usp=sharing](https://colab.research.google.com/drive/1BvTH0D75JxnbKIE5-5i5YaxXRPVCkUcT?usp=sharing)
- A basic linear regression algorithm to calculate the parameters (theta), using to fit the training data with least cost and be able to predict values for similar data (like testing data).
- Scaling used: Mean scaling with standard deviation
- Metric used:
  - Mean Squared Error (MSE) = 2794.58
- Input as :
  - *train(X_train, y_train, num_iter, alpha)*  
   X-train  --> Design matrix of features of trainig data  
   y_train  --> Target value vector of trainig data  
   num_iter --> number of iterations  
   alpha    --> learning rate  

  - *predict( X_test, y_test, y_train, theta)*  
   X-test   --> Design matrix of features of testing data  
   y_test   --> Target value vector of testing data  
   y_train  --> Target value vector of trainig data (for descaling)  
   theta    --> parameter obtained after training  
   
- Problems faced :
   - Introduction to Google Colab
   - Importing files into colab

- figures:
  - Cost vs iterations  
  ![Cost vs iterations](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/b2f32e017e77ddd276dafa2fb04929a58866c05d/Linear%20Cost.png)  
   
  - Hypothesis (line plot) and Target values (scatter points) of first 10 test examples  
  ![Hypothesis (line plot) and Target values (scatter points) of first 10 test examples](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/b2f32e017e77ddd276dafa2fb04929a58866c05d/Linear%20hypothesis.png)
  

### 1. Polynomial Regression:
- Google Colab : [https://colab.research.google.com/drive/1mCTIqBtZl719TIAQv4Z27fLv4dZt9ncm?usp=sharing](https://colab.research.google.com/drive/1mCTIqBtZl719TIAQv4Z27fLv4dZt9ncm?usp=sharing)
- Similar to linear regression, this algorithm calculates parameter and uses it to predict further values but also used additional higher degree terms (here upto 3rd degree) as additional features to better fit the data. Hence, regularisation is also used in order to prevent parameter from overfitting the training data.
- Scaling used: Mean scaling with standard deviation
- Metric used:
  - Mean Squared Error (MSE) = 61.64
- Input as :
  - *train(X_train, y_train, num_iter, alpha)*  
   X-train   --> Design matrix of features of trainig data  
   y_train   --> Target value vector of trainig data  
   num_iter  --> number of iterations  
   alpha     --> learning rate
   reg_coeff --> regularisation coefficient

  - *predict( X_test, y_test, y_train, theta)*  
   X-test   --> Design matrix of features of testing data  
   y_test   --> Target value vector of testing data  
   y_train  --> Target value vector of trainig data (for descaling)  
   theta    --> parameter obtained after training  
   
- Problems faced :
   - Appending higher degree terms in design matrix

- figures:
  - Cost vs iterations  
  ![Cost vs iterations](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/b2f32e017e77ddd276dafa2fb04929a58866c05d/Polynomial%20Cost.png)
  
  - Hypothesis (line plot) and Target values (scatter points) of first 10 test examples  
  ![Hypothesis (line plot) and Target values (scatter points) of first 10 test examples](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/b2f32e017e77ddd276dafa2fb04929a58866c05d/Polynomial%20hypothesis.png)
  

### 3. Logistic Regression:
- Google Colab : [https://colab.research.google.com/drive/1sAeQuDJRSB-FJrJp-2DUvTwnHSlQtYOL?usp=sharing](https://colab.research.google.com/drive/1sAeQuDJRSB-FJrJp-2DUvTwnHSlQtYOL?usp=sharing)
- Classification algorithm, using sigmoid as activation function to calculate probabilites of a data point (from EMNIST dataset) to belong in any of 26 classes (Alphabets)
- Scaling used: Mean scaling with standard deviation (if standard deviation for any feature, std = 0, then std is taken to be, std = 1)
- Metric used:
  - Log Loss = 0.67
  - Accuracy = 55.0 %
- Input as :
  - *train(X_train, y_train, num_iter, alpha)*  
   X-train   --> Design matrix of features of trainig data  
   y_train   --> Target value vector of trainig data  
   num_iter  --> number of iterations  
   alpha     --> learning rate  
   reg_coeff --> regularisation coefficient

  - *predict(X, y, theta, num_class)*  
  X           --> Design matrix of features of testing data  
  y           --> Target value vector of testing data  
  theta       --> parameter obtained after training  
  num_classes --> number of classes   
   
- Problems faced :
   - Converting class vector to probability matrix and vice versa

- figures:
  - Cost vs iterations  
  ![Cost vs iterations](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/29879f2d0ee7076407bf3b622ba9b2a9778e18da/Logistic%20cost.png)
 

### 4. K Nearest Neighbours (KNN):
- Google Colab : [https://colab.research.google.com/drive/1AdDIoRqzOJ_9KqfaFuZp0pgsirHjMxeM?usp=sharing](https://colab.research.google.com/drive/1AdDIoRqzOJ_9KqfaFuZp0pgsirHjMxeM?usp=sharing)
- Classifying data comparing data points from testing data to each of training data, finding K nearest neighbours (using eiclidian distance) and assigning datapoint to the class of the neighbour with highest mode (frequency)
- Metric used:
   - Accuracy = 84.18%
- Input as :
  - *test(X_test, y_test, X_train, y_train, num_classes, k=5)*  
  X_test --> Design matrix of test data  
  y_test --> target class vector  
  X_train --> Design matrix of training (reference) data  
  y_train --> reference class vector  
  K --> Number of neighbours  
   
- Problems faced :
   - Implementing algorithm
   - Long testing time

### 5. K - Means Clustering:
- Google Colab : [https://colab.research.google.com/drive/1_oKEYt5Rz8LO5LEnnR4Yizt3OGBwgxMl?usp=sharing](https://colab.research.google.com/drive/1_oKEYt5Rz8LO5LEnnR4Yizt3OGBwgxMl?usp=sharing)
- Clustering algorithm to group similar data points together, by calculating euclidian distance from centroid and updating centroid in each iteration.
- Metric used:
  - Dunn Index = 0.2244915204008792
- Input as :
  - *Kmeans (X, num_iter, k)*  
  X        --> mxn design matrix of data points with n features and m points  
  num_iter --> number of iterations  
  k        --> number of clusters   
   
- Problems faced :
   - Assigning cluster to a data point i.e., segregating data points into clusters

- Scope of improvement
  - Use 3-D array or nested list/array as alternate approach to store clusters 
  - Use K++ to choose initial centroids

### 6. Neural Network:
- Implemented a 2 layer neural network (1-hidden and 1-output) to perform classification, linear regression and polynomial regression  
- ### A. Classification
    - Google Colab : [https://colab.research.google.com/drive/1OGb-ZdJtv7gi367qsgdKP4gTpfWOpa_U?usp=sharing](https://colab.research.google.com/drive/1OGb-ZdJtv7gi367qsgdKP4gTpfWOpa_U?usp=sharing)
    - Sigmoid used as activation function
    - Metric used:  
      - Accuracy = 73.08 %  
    - Input as :
      - *NNClassificationTrain(X, y, num_classes, num_iter, alpha, reg_coeff)*  
      X             --> Design matrix of training examples  
      y             --> Training class vector  
      num_classes   --> number of classes  
      num_iter      --> number of iterations  
      alpha         -->learning rate  
      reg_coeff     --> regularisation coefficiet  
      
      - *NNClassificationTest(X, y, theta1, theta2)*  
      X        --> Design matrix of features of testing data  
      y        --> Target value vector of testing data  
      theta1   --> parameter for input layer to hidden layer obtained after training  
      theta2   --> parameter for hidden layer to output layer obtained after training  

    - Problems faced :
      - Implementing back propogation
      - Adjusting learning rate and regularization coefficient

    - figures:
      - Cost vs iterations  
      ![Cost vs iterations](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/29879f2d0ee7076407bf3b622ba9b2a9778e18da/NN%20Classification%20Cost.png)
      
    - Scope of improvement  
      - Increase layers and/or adjust the number of units in hidden layer(s)

- ### B. Linear Regression
    - Google Colab : [https://colab.research.google.com/drive/1OkAJJeCr_hRzDYO2Tci7DF_Vq0ee0352?usp=sharing](https://colab.research.google.com/drive/1OkAJJeCr_hRzDYO2Tci7DF_Vq0ee0352?usp=sharing)
    - Scaling used: Mean scaling with standard deviation
    - Metric used:  
      - Root Mean Squared Error (RMSE) = 61.75
    - Input as :
      - *NNLinearTrain(X, y, num_iter, alpha, reg_coeff)*  
      X             --> Design matrix of training examples  
      y             --> target vector  
      num_iter      --> number of iterations  
      alpha         --> learning rate  
      reg_coeff     --> regularisation coefficient  
      
      - *NNLinearTest(X_test, y_test, y_train, theta1, theta2)*
      X_test   --> Design matrix of features of testing data  
      y_test   --> Target value vector of testing data  
      y_train  --> Target value vector of trainig data (for descaling)  
      theta1   --> parameter for input layer to hidden layer obtained after training  
      theta2   --> parameter for hidden layer to output layer obtained after training  

    - figures:
      - Cost vs iterations  
      ![Cost vs iterations](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/b2f32e017e77ddd276dafa2fb04929a58866c05d/NN%20Linear%20cost.png)

- ### C. Polynomial Regression
    - Google Colab : [https://colab.research.google.com/drive/10URIruEMapPYIC-OtKb2lQuSK1ON3qLG?usp=sharing](https://colab.research.google.com/drive/10URIruEMapPYIC-OtKb2lQuSK1ON3qLG?usp=sharing)
    - Scaling used: Mean scaling with standard deviation
    - Metric used: 
      - Root Mean Squared Error (RMSE) = 15.39
    - Input as :
      - *NNPolynomialTrain(X, y, num_iter, alpha, reg_coeff)*  
      X             --> Design matrix of training examples  
      y             --> target vector  
      num_iter      --> number of iterations  
      alpha         --> learning rate  
      reg_coeff     --> regularisation coefficient  
      
      - *NNPolynomialTest(X_test, y_test, y_train, theta1, theta2)*  
      X_test   --> Design matrix of features of testing data  
      y_test   --> Target value vector of testing data  
      y_train  --> Target value vector of trainig data (for descaling)  
      theta1   --> parameter for input layer to hidden layer obtained after training  
      theta2   --> parameter for hidden layer to output layer obtained after training  

    - figures:
      - Cost vs iterations  
      ![Cost vs iterations](https://github.com/Das-Arijit/Cyberlib---ML-Bootcamp/blob/b2f32e017e77ddd276dafa2fb04929a58866c05d/NN%20polynomial%20cost.png)
      
