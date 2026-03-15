import numpy as np

# Loss Functions
# Parameters:
# loss: Type of loss function: MSE, MAE, ErrorRate, accuracy, BCE

class Estimate_Loss:
  def __init__(self, loss = 'MSE'):
    self.loss = loss

  # To select the loss function with respect to the input
  def update(self, Y_pred, Y):

    if self.loss == "MSE":
      return self.meanSquareError(Y_pred, Y)

    elif self.loss == "MAE":
      return self.meanAbsoluteError(Y_pred, Y)

    elif self.loss == "ErrorRate":
      return self.ErrorRate(Y_pred, Y)

    elif self.loss == "accuracy":
      return self.accuracy(Y_pred, Y)

    elif self.loss == "BCE":
      return self.BinaryCrossEntropy(Y_pred, Y)

    else:
      raise ValueError(f"{self.loss} loss function is not defined")

  # Mean of the square difference between actual and predicted value
  # Used for regression task
  def meanSquareError(self, Y_pred, Y):

    loss = np.mean((Y - Y_pred)**2)
    return loss

  # Mean of the absolute difference between actual and predicted value
  # Used for regression task
  def meanAbsoluteError(self, Y_pred, Y):
    loss = np.mean(np.abs(Y - Y_pred))
    return loss

  # Tell percentage of wrong predictions
  # Used for classification task
  def ErrorRate(self, Y_pred, Y):
    y_predict = np.round(Y_pred)
    loss = np.mean((Y != y_predict))
    return loss

  # Tell percentage of correct predictions
  # Used for classification task
  def accuracy(self, Y_pred, Y):
    y_predict = np.round(Y_pred)
    loss = np.mean(Y == y_predict)
    return loss

  # Used for classification task
  def BinaryCrossEntropy(self, Y_pred, Y):
    N = Y.shape[0]
    loss = -np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))
    return loss