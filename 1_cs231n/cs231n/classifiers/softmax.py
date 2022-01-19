import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = W.shape[0] # num of classes
  N = X.shape[1] # num of samples
  f = np.matmul(W, X) # C by N
  f -= np.amax(f, axis=0) # C by N
  p = np.exp(f) / np.sum(np.exp(f), axis=0) # C by N
  loss = np.mean(-np.log(p[y, np.arange(N)] + 1e-6)) + 0.5 * reg * np.sum(W*W)
  # compute grads
  dscores = np.copy(p)
  dscores[y, np.arange(N)] -= 1
  dscores /= N
  dW = np.dot(dscores, X.T)
  dW += reg*W 
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
