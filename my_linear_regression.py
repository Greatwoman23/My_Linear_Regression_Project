# Import necessary libraries
import numpy as np
import statistics
import matplotlib.pyplot as plt
import math

# Generate random data
X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)
X_1 = X.reshape(1, -1)[0]
Y_1 = y.reshape(1, -1)[0]

# Define the hypothesis function
def h(x, theta_):
    return np.dot(x, np.array(theta_)).reshape(-1, 1)

# Create a class for Least Squares Regression
class LeastSquaresRegression():
    def __init__(self):
        self.theta_ = None

    # Fit the model to the data
    def fit(self, X, y):
        # Compute the means of X and y
        X_mean = statistics.mean(X_1)
        Y_mean = statistics.mean(Y_1)

        # Calculate the deviations from the means
        all_x = [i - X_mean for i in X_1]
        all_y = [i - Y_mean for i in Y_1]

        # Calculate the Pearson correlation score
        all_xy = sum([all_x[i] * all_y[i] for i in range(len(all_x))])
        sum_squared_error_x = sum([i ** 2 for i in all_x])
        sum_squared_error_y = sum([i ** 2 for i in all_y])
        E_sum = math.sqrt(sum_squared_error_x * sum_squared_error_y)
        person_corr_score = all_xy / E_sum

        # Calculate the standard deviations of X and y
        Sy = math.sqrt((sum_squared_error_y) / (len(Y_1) - 1))
        Sx = math.sqrt((sum_squared_error_x) / (len(X_1) - 1))

        # Calculate the slope and y-intercept of the regression line
        slope_line = person_corr_score * (Sy / Sx)
        Y_intercept = Y_mean - slope_line * X_mean

        # Store the coefficients in theta_
        self.theta_ = np.array([Y_intercept, slope_line]).reshape(-1, 1)
        return self.theta_

    # Make predictions using the fitted model
    def predict(self, X):
        return h(X, self.theta_)

# Function to plot the data and the regression line
def my_plot(X, y, y_new):
    plt.title("LeastSquaresRegression")
    plt.scatter(X, y, c=y, cmap="viridis")
    plt.colorbar()
    plt.plot(X, y_new, "r")
    plt.xlabel("X-axis")
    plt.ylabel("Y-pred")
    plt.grid()
    

# Add a bias column to the feature matrix X
def bias_column(X):
    new = np.ones((len(X_1), 1))
    X = np.append(new, X, axis=1)
    return X

# Preprocess the data with a bias column
X_new = bias_column(X)

# Create and fit the regression model
model = LeastSquaresRegression()
model.fit(X, y)

# Make predictions using the fitted model
y_predictions = model.predict(X_new)

# Define a function to compute the mean squared error
def mean_squared_error(y_predicted, y_label):
    MSE = sum([(y_label[i] - y_predicted[i]) ** 2 for i in range(len(y_label))]) / len(y_label)
    return MSE

# Define two functions representing a quadratic function and its derivative
def f(x):
    a = np.array([[2], [6]])
    all = 3 + np.dot((x - a).reshape(1, -1), (x - a))
    return all

def fprime(x):
    a = np.array([[2], [6]])
    return 2 * (x - a).reshape(-1,1)

# Test the GradientDescentOptimizer class
def test_class_gradient_descent_optimizer(self):
    gdo = eg.GradientDescentOptimizer(TestAgent.f, TestAgent.f_prime, np.random.normal(size=(2,)), 0.1)
    gdo.optimize(10)
    user_values = gdo.getCurrentValue()
    self.assertTrue(1.5 < user_values[0])
    self.assertTrue(user_values[0] < 1.9)
    self.assertTrue(5.0 < user_values[1])
    self.assertTrue(user_values[1] < 5.5)

# Class for Gradient Descent Optimizer
class GradientDescentOptimizer():

    def __init__(self, f, fprime, start, learning_rate=0.1):
        self.f_ = f                       # The function to optimize
        self.fprime_ = fprime             # The derivative of the function
        self.current_ = start.reshape(-1, 1)   # Current value of the optimization parameter
        self.learning_rate_ = learning_rate    # Learning rate for gradient descent

        self.history_ = [start]          # History of parameter values during optimization

    # Perform a single optimization step using gradient descent
    def step(self):
        gradient = self.fprime_(self.current_)
        self.current_ -= self.learning_rate_ * gradient
        self.history_.append(self.current_)

    # Optimize the function for a given number of iterations
    def optimize(self, iterations=100):
        for i in range(iterations):
            self.step()

    # Get the current value of the optimization parameter
    def getCurrentValue(self):
        return self.current_

    # Print the result of the optimization
    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))
