import numpy as np

# Parameters:
# alpha: Learning rate defined to learn the model
# n_iterations: Number of iteration
# epsilon: Stopping criteria of gradient descent usually very small i.e. 1e-8
# beta: Parameter for momentum and rmsprop, usually 0.9
# beta1 & beta2: Parameters for adam optimizer, usually 0.9 and 0.99 respectively
# batch_size: To estimate the size of mini batches for SGD, momentum, rmsprop and adam

class MLP_Optimizer:
  def __init__(self, alpha=0.01, n_iteration=100, optimizer='SGD', epsilon=1e-8, beta=0.9, beta1=0.9, beta2=0.99, batch_size=8):
    self.alpha = alpha
    self.n_iteration = n_iteration
    self.optimizer = optimizer
    self.epsilon = epsilon
    self.beta = beta
    self.beta1 = beta1
    self.beta2 = beta2
    self.batch_size = batch_size
    self.V = {}  # Momentum parameter for each layer
    self.S = {}  # RMSProp parameter for each layer
    self.t = 0   # Timestep for Adam optimizer

  def initialize_moments(self, theta, layer_name):
    self.V[layer_name] = np.zeros_like(theta)  # Initialize momentum (V) for the layer
    self.S[layer_name] = np.zeros_like(theta)  # Initialize RMSProp (S) for the layer

  # The following Optimizers [Adam, SGD, Momentum, RMSProp] use mini-batches to update weights 
  def mini_batch(self, X, Y): 
    N = X.shape[0]
    ind = np.arange(N)
    # Before each epoch the data is shuffled and then mini batches are created
    np.random.shuffle(ind)
    X_s = X[ind]
    Y_s = Y[ind]
    mini_batch = []

    for i in range (0, N, self.batch_size):
      end = i + self.batch_size
      X_batch = X_s[i:end]
      Y_batch = Y_s[i:end]
      mini_batch.append((X_batch, Y_batch))
    
    return mini_batch

  def update(self, theta, grad, layer_name):
    if self.optimizer == "SGD":
      return self.SGD(theta, grad, layer_name)
    elif self.optimizer == "momentum":
      return self.momentum(theta, grad, layer_name)
    elif self.optimizer == "rmsprop":
      return self.rmsprop(theta, grad, layer_name)
    elif self.optimizer == "adam":
      return self.Adam(theta, grad, layer_name)
    else:
      raise ValueError(f"{self.optimizer} optimizer is not defined")

  def SGD(self, theta, grad, layer_name):
    theta -= self.alpha * grad  # Update theta w.r.t the computed gradient: θ = θ − α ∇(J(θ))
    return theta

  def momentum(self, theta, grad, layer_name):
    self.V[layer_name] = self.beta * self.V[layer_name] + (1 - self.beta) * grad  # Update in momentum w.r.t the computed gradient: V(θ) = β1 V(θ) + (1 - β1) ∇(J(θ)) | β1 = 0.9
    theta -= self.alpha * self.V[layer_name]
    return theta

  def rmsprop(self, theta, grad, layer_name):
    self.S[layer_name] = self.beta * self.S[layer_name] + (1 - self.beta) * (grad ** 2)  # Update for RMSProp w.r.t the computed gradient: S(θ) = β2 S(θ) + (1 - β2)(J(θ) ^ 2)   [element wise square]
    theta -= self.alpha * grad / (np.sqrt(self.S[layer_name]) + self.epsilon)
    return theta

  def Adam(self, theta, grad, layer_name):
    self.V[layer_name] = self.beta1 * self.V[layer_name] + (1 - self.beta1) * grad   # Update in momentum w.r.t the computed gradient: V(θ) = β1 V(θ) + (1 - β1) ∇(J(θ)) | β1 = 0.9
    self.S[layer_name] = self.beta2 * self.S[layer_name] + (1 - self.beta2) * (grad ** 2)  # Update for RMSProp w.r.t the computed gradient: S(θ) = β2 S(θ) + (1 - β2)(J(θ) ^ 2)   [element wise square]

    V_corr = self.V[layer_name] / (1 - self.beta1 ** (self.t + 1))  # Corrected V(θ): V_corr(θ) = V / (1 - β1 ^ t)
    S_corr = self.S[layer_name] / (1 - self.beta2 ** (self.t + 1))  # Corrected S(θ): S_corr(θ) = S / (1 - β2 ^ t)

    theta -= self.alpha * V_corr / (np.sqrt(S_corr) + self.epsilon) # Update theta: θ = [θ − α ∇(V_corr(θ))] / [sqrt(S_corr(θ)) + e]
    self.t += 1
    return theta
