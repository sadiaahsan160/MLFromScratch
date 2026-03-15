import numpy as np
from Loss_Functions.Losses import Estimate_Loss

# Multi Layer Perceptron
# Parameters:
# input_size: Size of input Layer
# hidden_size: Size of Hidden Layer
# output_size: Size of Output Layer [2 if Classification task otherwise 1]
# Optimizer: To update theta
# n_iterations: Number of iteration
# loss function: To estimate losses


class MLP:
  def __init__(self, input_size, hidden_size, output_size, optimizer=None, lossfunction='BCE'):
    
    self.w_hi = np.random.rand(input_size, hidden_size)  # input to hidden layer weights w.r.t the size of hidden of input layer
    self.b1 = np.zeros((1, hidden_size))  # hidden layer bias
    self.w_oh = np.random.rand(hidden_size, output_size)  # hidden to output layer weights w.r.t the size of hidden and output layer
    self.b2 = np.zeros((1, output_size))  # output layer bias
        
    # Optimizer and loss function
    self.optimizer = optimizer
    self.lossfunction = Estimate_Loss(loss=lossfunction) # Defining Loss Function
        
    # Initialize optimizer's moments for both sets of weights
    self.optimizer.initialize_moments(self.w_hi, 'w_hi') # Initializing weights of input to hidden layer
    self.optimizer.initialize_moments(self.w_oh, 'w_oh') # Initializing weights of hidden to output layer
    self.optimizer.initialize_moments(self.b1, 'b1')  # Initializing bias of hidden layer
    self.optimizer.initialize_moments(self.b2, 'b2')  # Initializing bias of output layer

  def sigmoid(self, z):
    sig = 1 / (1 + np.exp(-z))  # Sigmoid Function
    return sig

  def deriv_sigmoid(self, z):
    d_sig = z * (1 - z)  # Derivative of Sigmoid Function
    return d_sig

  def forward(self, X):
    # Forward pass from input to output layer
    self.hidden_layer = self.sigmoid(np.dot(X, self.w_hi) + self.b1)  # Input to hidden layer [input layer * weights of input to hidden layer with sigmoid function to introduce noon linearity]
    self.y_pred = self.sigmoid(np.dot(self.hidden_layer, self.w_oh) + self.b2)  # Hidden to output layer [hidden layer * weights of hidden to output layer with sigmoid function to introduce noon linearity]
    return self.y_pred  # Final predicted output

  def back_propagation(self, X, Y):
    # To update the weights of model w.r.t error computed
    N = X.shape[0]
        
    error = self.y_pred - Y  # Error
        
    # Gradients for weights and biases from hidden to output  layer
    dw_oh = np.dot(self.hidden_layer.T, error) / N  # Derivative w.r.t weights of hidden to output layer
    db2 = np.sum(error, axis=0, keepdims=True) / N
        
    # Gradients for weights and biases from input to hidden layer
    d_h = np.dot(error, self.w_oh.T) * self.deriv_sigmoid(self.hidden_layer)

    dw_hi = np.dot(X.T, d_h) / N  # Derivative w.r.t weights of input to hidden layer
    db1 = np.sum(d_h, axis=0, keepdims=True) / N
        
    # Updating both the weights
    self.w_hi = self.optimizer.update(self.w_hi, dw_hi, 'w_hi')  # Update input to hidden layer weights
    self.b1 = self.optimizer.update(self.b1, db1, 'b1')        # Update hidden layer bias
    self.w_oh = self.optimizer.update(self.w_oh, dw_oh, 'w_oh')  # Update hidden tooutput layer weights
    self.b2 = self.optimizer.update(self.b2, db2, 'b2')        # Update output layer bias


  def best_fit(self, X, Y):
    losses = []
    for i in range(self.optimizer.n_iteration):
      mini_batches = self.optimizer.mini_batch(X, Y)  # Here passing the mini batches for the optimizers [Adam, RMSProp, momentum and SGD] 
      # Applying forward and backpropagation on mini batches 
      for X_batch, Y_batch in mini_batches:
        # Forward pass to compute predictions
        Y_pred = self.forward(X_batch)
        
      # Compute loss
      loss = self.lossfunction.update(Y_pred, Y_batch)
      losses.append(loss)
        
      # Backpropagate to update weights w.r.t error
      self.back_propagation(X_batch, Y_batch)
        
    return losses