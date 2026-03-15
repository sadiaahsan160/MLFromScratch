import numpy as np
from Loss_Functions.Losses import Estimate_Loss

# Non Linear Classification
# Parameters:
# Optimizer: To update theta
# loss function: To estimate losses
# K: Degree of Polynomial

class Nonlinear_classification:
  def __init__(self, optimizer=None, lossfunction='BCE', K=2):
    self.optimizer = optimizer
    self.lossfunction = Estimate_Loss(loss=lossfunction)
    self.theta = None
    self.K = K

  # Function to map feature X into non linear feature ϕ(x)
  # K: Degree of Polynomial
  def transformation(self, X):
    N, F = X.shape
    # Initialize Phi
    Phi = np.ones((N, 1))
    # Loop over each degree from 1 to K
    for i in range(1, self.K + 1):
      # Generate combinations of features for the current degree
      for j in range(F):
        Phi = np.hstack((Phi, X[:, j:j+1]**i)) 
    return Phi

  # Definig sigmoid function for classification task
  def sigmoid(self, z):
    sig = 1 / (1 + np.exp(-z))  
    return sig

  # Fitting the model to find the best theta (weights) using the training data and corresponding labels X and Y
  # There is a linear function: y(ϕ(x)) = θ0 + θ1.ϕ(x)
  def best_fit(self, x, Y):
    X = self.transformation(x)
    N = X.shape[0]
    Y = Y.ravel()
    # Initially theta is initialized with 0 of length equal to the number of features in training data ϕ(x)
    self.theta = np.zeros(X.shape[1])
    # Definig the optimizer for the classification task
    self.optimizer.initialize_moments(self.theta)
    losses = []

    # Optimization process
    for i in range(self.optimizer.n_iteration):
      self.theta = self.optimizer.update(self.theta, X, Y)
      # Using the current theta for prediction
      Y_pred = self.predict(x)
      # Computing loss at each epoch/iteration
      loss = self.lossfunction.update(Y_pred, Y)
      losses.append(loss)

    return self.theta, losses

  # For making predictions 
  def predict(self, x):
    X = self.transformation(x)
    Y_pred = self.sigmoid(np.dot(X, self.theta))
    return Y_pred

  # For predicting classes
  def predict_class(self, x):
    Y_pred_class = self.predict(x)
    return np.where(Y_pred_class >= 0.5, 1, 0)