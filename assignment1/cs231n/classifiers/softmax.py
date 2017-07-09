import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in xrange(num_train):
    score = X[i].dot(W)
    score -= np.max(score)
    exp_score = np.exp(score)
    loss += -score[y[i]] + np.log(np.sum(exp_score))

    for j in xrange(num_class):
      score_naive = exp_score[j] / np.sum(exp_score)
      if j == y[i]:
        dW[:, j] += (score_naive - 1) * X[i]
      else:
        dW[:, j] += score_naive * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW = dW / num_train + reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  num_train = X.shape[0]
  num_class = W.shape[1]
  score = X.dot(W) # (N,C)
  score -= np.max(score,axis=1).reshape(-1,1)
  
  exp_score = np.exp(score)
  softmax_score = exp_score / np.sum(exp_score, axis=1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_score[np.arange(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  softmax_score[np.arange(num_train), y] -= 1
  dW = X.T.dot(softmax_score)
  dW = dW/num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

