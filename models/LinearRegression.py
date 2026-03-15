import numpy as np
from Loss_Functions.Losses import Estimate_Loss

# Linear Regression
# Parameters:
# Optimizer: To update theta
# n_iterations: Number of iteration
# loss function: To estimate losses

class linear_regression:
  def __init__(self, optimizer=None, lossfunction = 'MSE'):
    self.optimizer = optimizer
    self.lossfunction = Estimate_Loss(loss=lossfunction)
    self.theta = None

  # Fitting the model to find the best theta (weights) using the training data and corresponding labels X and Y
  # There is a linear function: y(x) = θ0 + θ1.x

  def best_fit(self, x, Y):

    N = x.shape[0]
    # Since x0 is always 1 so an additional coloumn of length N is added to the input data using concatenation function
    # Finally X will be a matrix of size N x 2 where N is the dimensionality
    X = np.concatenate((np.ones((N, 1)), x), axis=1)

    Y = Y.ravel()

    # Initially theta is initialized with 0 of length equal to the number of features in training data X
    self.theta = np.zeros(X.shape[1])

    # Definig the optimizer for the regression task
    self.optimizer.initialize_moments(self.theta)
    losses = []

    # Optimization process
    for i in range(self.optimizer.n_iteration):

      # Updating the weights (theta) at each epoch
      self.theta = self.optimizer.update(self.theta, X, Y)

      # Computing loss at each epoch/iteration
      # Using the current theta for prediction
      Y_pred = self.predict(x)
      loss = self.lossfunction.update(Y_pred, Y)
      losses.append(loss)

    return self.theta, losses

# For making predictions by taking unseen data
  def predict(self, x):

    N = x.shape[0]
    X = np.concatenate((np.ones((N, 1)), x), axis=1)
    Y_predict = np.dot(X, self.theta) # Dot product of new data and computed theta

    return Y_predict