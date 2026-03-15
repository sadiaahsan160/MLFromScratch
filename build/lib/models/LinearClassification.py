import numpy as np
from Loss_Functions.Losses import Estimate_Loss

# Linear Classification
# Parameters:
# n_iterations: number of iteration

class linear_classification:
  def __init__(self, optimizer=None, lossfunction = 'accuracy'):
    self.optimizer = optimizer
    self.lossfunction = Estimate_Loss(loss=lossfunction)
    self.theta = None


  def sigmoid(self, z):
    sig = 1 / (1 + np.exp(-z))
    return sig

  # Fitting the model to find the best theta (weights) using the training data and corresponding labels X and Y
  # There is a linear function: y(x) = θ0 + θ1.x

  def best_fit(self, x, Y):

    N = x.shape[0]
    # Since x0 is always 1 so an additional coloumn of length N is added to the input data using concatenation function
    # Finally X will be a matrix of size N x 2
    X = np.concatenate((np.ones((N, 1)), x), axis=1)

    Y = Y.ravel()

    # Initially theta is initialized with 0 of length equal to the number of features in training data X
    self.theta = np.zeros(X.shape[1])
    
    losses = []

    # Optimization process
    for i in range(self.optimizer.n_iteration):
      # Sigmoid to find the optimal values of θ
      # Using the current theta for prediction
      Y_pred = self.sigmoid(np.dot(X, self.theta))

      # J(θ) is the cost function
      # ∇(J(θ)) is the gradient (derivative) of J wrt to θ
      # grad = (X.T @ (Y_pred - Y))/N    # Formula for gradient descent ∇(J(θ))

      self.optimizer.initialize_moments(self.theta)

      # Updating the weights (theta) at each epoch
      self.theta = self.optimizer.update(self.theta, X, Y)
      # Definig the optimizer for the classification task
      

      # Computing loss at each epoch/iteration
      loss = self.lossfunction.update(Y_pred, Y)
      losses.append(loss)

    return self.theta, losses

# For making predictions by taking unseen data
  def predict(self, x):

    N = x.shape[0]
    X = np.concatenate((np.ones((N, 1)), x), axis=1)
    Y_predict = self.sigmoid(np.dot(X, self.theta)) # Dot product of new data and computed theta

    return Y_predict

# For predicting classes
  def predict_class(self, x):
    Y_predict = self.predict(x)
    return np.array([1 if i > 0.5 else 0 for i in Y_predict])