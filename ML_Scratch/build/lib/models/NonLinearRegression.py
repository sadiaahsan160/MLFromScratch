import numpy as np
from Loss_Functions.Losses import Estimate_Loss

# Non Linear Regression
# Parameters:
# Optimizer: To update theta
# n_iterations: Number of iteration
# loss function: To estimate losses

class Nonlinear_regression:
  def __init__(self, optimizer=None, lossfunction = 'MSE', K = 2):
    self.optimizer = optimizer
    self.lossfunction = Estimate_Loss(loss=lossfunction)
    self.theta = None
    self.K = K

  # Function to map feature X into non linear feature ϕ(x)
  # K: Degree of Polynomial
  def transformation(self, X):

    X = X.flatten()
    N = X.shape[0]
    #initialize Phi
    Phi = np.zeros((N, self.K+1))

    # Compute the feature matrix in stages
    for i in range(self.K+1):
        Phi[:, i] = X ** i

    return Phi


  # Fitting the model to find the best theta (weights) using the training data and corresponding labels X and Y
  # There is a linear function: y(x) = θ0 + θ1.x

  def best_fit(self, x, Y):

    X = self.transformation(x) # Mapping the the low dimension data to high dimentionality 
    N = X.shape[0]
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

    X = self.transformation(x)
    N = X.shape[0]
    Y_predict = np.dot(X, self.theta) # Dot product of new data and computed theta

    return Y_predict