# Welcome to My Linear Regression
***

## Task
The task is to implement multiple functions and 2 classes:
- def h(x, theta)
Write the linear hypothesis function. (see above)

- def mean_squared_error(y_pred, y_label)
Write the Mean Squared Error function between the predicted values and the labels.

- def bias_column(x)
Write a function which adds one to each instance.

X_new = bias_column(x)
print(X[:5])
print(" ---- ")
print(X_new[:5])

Classes
class LeastSquaresRegression: (see description above)
  def __init__(self, )
  def fit()
  def predict

class GradientDescentOptimizer: (see description above)
  def __init__()
  def step()
  def optimize()
  def getCurrentValue()

## Description
In this project, I have implemented a Least Squares Regression model and a Gradient Descent Optimizer using Python.
The Least Squares Regression model is used for linear regression, while the Gradient Descent Optimizer is a numerical optimization algorithm that can be used to find the minimum of a given function.

## Installation
The following libraries were installed:
numpy
statistics
matplotlib

## Usage
Least Squares Regression:
The LeastSquaresRegression class is used to perform linear regression using the least squares method. 
It fits a linear model to the given data and makes predictions using the model.

Gradient Descent Optimizer:
The GradientDescentOptimizer class is used to optimize a given function using gradient descent. 
It performs a series of optimization steps to find the minimum of the function.
```
./ python my_linear_regression.py
```

### The Core Team
deniran_o


<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px'></span>
