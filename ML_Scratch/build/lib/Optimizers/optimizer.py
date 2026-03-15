import numpy as np

# Optimizers
# Parameters:
# alpha: Learning rate defined to learn the model
# n_iterations: Number of iteration
# epsilon: Stopping criteria of gradient descent usually very small i.e. 1e-8
# beta: Parameter for momentum and rmsprop, usually 0.9
# beta1 & beta2: Parameters for adam optimizer, usually 0.9 and 0.99 respectively
# batch_size: To estimate the size of mini batches for SGD, momentum, rmsprop and adam

class Optimizer:
  def __init__(self, alpha = 0.01, n_iteration = 100, optimizer = 'GradientDescent', epsilon = 1e-8, beta = 0.9, beta1 = 0.9, beta2 = 0.99, batch_size = 8):
    self.alpha = alpha
    self.n_iteration = n_iteration
    self.optimizer = optimizer
    self.epsilon = epsilon
    self.beta = beta
    self.beta1 = beta1
    self.beta2 = beta2
    self.batch_size = batch_size
    self.S = None
    self.V = None
    self.t = 0

  def initialize_moments(self, theta):
    self.S = np.zeros_like(theta)
    self.V = np.zeros_like(theta)

  # To select the optimizer with respect to the input
  def update(self, theta, X, Y):

    if self.optimizer == "GradientDescent":
      return self.GradientDescent(theta, X, Y)

    elif self.optimizer == "SGD":
      return self.SGD(theta, X, Y)

    elif self.optimizer == "momentum":
      return self.momentum(theta, X, Y)

    elif self.optimizer == "rmsprop":
      return self.rmsprop(theta, X, Y)

    elif self.optimizer == "adam":
      return self.Adam(theta, X, Y)

    else:
      raise ValueError(f"{self.optimizer} optimizer is not defined")

  def GradientDescent(self, theta, X, Y):

    N = len(Y)
    i = 0
    while i < self.n_iteration:
      grad = (X.T @ (X @ theta - Y))/N    # Formula for gradient descent ∇(J(θ))
      theta -= self.alpha * grad    # Update of theta w.r.t the computed gradient: θk = θk−1 − α ∇(J(θ))

      if np.linalg.norm(grad) < self.epsilon:   # Stopping criteria: Stop if |∇(J(θ))| < e
        break
      i += 1

    return theta

  def SGD(self, theta, X, Y):

    # Mini batches are optimized in SGD
    # Each mini batch data is shuffeled before optimization

    N = len(Y)
    for epoch in range (self.n_iteration):
      ind = np.arange(N)
      np.random.shuffle(ind)
      X_s = X[ind]
      Y_s = Y[ind]

      for i in range (0, N, self.batch_size):
        end = i + self.batch_size
        X_batch = X_s[i:end]
        Y_batch = Y_s[i:end]
        grad = (X_batch.T @ (X_batch @ theta - Y_batch))/Y_batch.size
        theta -= self.alpha * grad    # Update theta w.r.t the computed gradient: θ = θ − α ∇(J(θ))
        grad = np.clip(grad, -1e3, 1e3) 

    return theta


  def momentum(self, theta, X, Y):

    # Compute gradient on mini batch
    # Each mini batch data is shuffeled before optimization

    N = len(Y)
    for epoch in range (self.n_iteration):
      ind = np.arange(N)
      np.random.shuffle(ind)
      X_s = X[ind]
      Y_s = Y[ind]

      for i in range (0, N, self.batch_size):

        end = i + self.batch_size
        X_batch = X_s[i:end]
        Y_batch = Y_s[i:end]
        grad = (X_batch.T @ (X_batch @ theta - Y_batch))/Y_batch.size # Gradient computation on current mini batch: ∇(J(θ))
        self.V = self.beta * self.V + (1 - self.beta) * grad  # Update in momentum: V(θ) = β V(θ) + (1 - β) ∇(J(θ)) | β = 0.9
        theta -= self.alpha * self.V    # Update theta w.r.t the computed gradient: θ = θ − α V(θ)

    return theta


  def rmsprop(self, theta, X, Y):

    # Compute gradient on mini batch
    # Each mini batch data is shuffeled before optimization

    N = len(Y)
    for epoch in range (self.n_iteration):
      ind = np.arange(N)
      np.random.shuffle(ind)
      X_s = X[ind]
      Y_s = Y[ind]

      for i in range (0, N, self.batch_size):
        end = i + self.batch_size
        X_batch = X_s[i:end]
        Y_batch = Y_s[i:end]
        grad = (X_batch.T @ (X_batch @ theta - Y_batch))/Y_batch.size  #  Gradient computation on current mini batch: ∇(J(θ))
        self.S = self.beta * self.S + (1 - self.beta) * (grad**2)  #  Update for RMSProp: S(θ) = β S(θ) + (1 - β)(J(θ))^2   [element wise square]
        theta -= self.alpha * grad / (np.sqrt(self.S) + self.epsilon)     #  Update theta w.r.t the computed gradient: θ = [θ − α ∇(J(θ))] / [sqrt(S(θ)) + e]
    return theta


  def Adam(self, theta, X, Y):

    # Compute gradient on mini batch
    # Each mini batch data is shuffeled before optimization

    N = len(Y)

    for epoch in range (self.n_iteration):
      ind = np.arange(N)
      np.random.shuffle(ind)
      X_s = X[ind]
      Y_s = Y[ind]
      # self.initialize_moments(theta)

      for i in range (0, N, self.batch_size):
        end = i + self.batch_size
        X_batch = X_s[i:end]
        Y_batch = Y_s[i:end]
        grad = (X_batch.T @ (X_batch @ theta - Y_batch))/Y_batch.size # Gradient computation on current mini batch: ∇(J(θ))

        self.V = self.beta1 * self.V + (1 - self.beta1) * grad  # Update in momentum: V(θ) = β V(θ) + (1 - β) ∇(J(θ)) | β = 0.9
        self.S = self.beta2 * self.S + (1 - self.beta2) * (grad**2)  #  Update for RMSProp: S(θ) = β S(θ) + (1 - β)(J(θ))^2   [element wise square]

        V_corr = self.V / (1 - self.beta1 ** (self.t+1))
        S_corr = self.S / (1 - self.beta2 ** (self.t+1))
        theta -= self.alpha * V_corr / (np.sqrt(S_corr) + self.epsilon)     # Update theta w.r.t the computed gradient: θ = [θ − α ∇(J(θ))] / [sqrt(S(θ)) + e]

        self.t += 1

    return theta