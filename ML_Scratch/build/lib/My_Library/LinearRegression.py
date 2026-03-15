import numpy as np

# Linear Regression 
# Parameters:
# alpha: learning rate defined to learn the model 
# n_iterations: number of iteration 
# epsilon: Stopping criteria of gradient descent

class linear_regression:
  def __init__(self, alpha = 0.001, n_iterations = 10, epsilon = 0.0001):
    self.alpha = alpha
    self.n_iterations = n_iterations
    self.epsilon = epsilon
    self.theta = None
    self.bias = None

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

    i=0 # Initial iteration

    # Gradient Descent to find the optimal values of θ 
    # It is repeated until the model converges
    # θk = θk−1 − α ∇(J(θ)) | θk−1: current theta | θk: new theta | α: learning rate (already defined)
    # J(θ) is the cost function 
    # ∇(J(θ)) is the gradient (derivative) of J wrt to θ

    while i < self.n_iterations:

      grad = (X.T @ (X @ self.theta - Y))/N    # Formula for gradient descent
      self.theta -= self.alpha * grad    # Update of theta w.r.t the computed gradient: θk = θk−1 − α ∇(J(θ))

      if np.linalg.norm(grad) < self.epsilon:   # Stopping criteria: Stop if |∇(J(θ))| < e
        break

      i += 1

    return self.theta

# For making predictions by taking unseen data 
  def predict(self, x):

    N = x.shape[0]
    X = np.concatenate((np.ones((N, 1)), x), axis=1)
    Y_predict = np.dot(X, self.theta) # Dot product of new data and computed theta

    return Y_predict