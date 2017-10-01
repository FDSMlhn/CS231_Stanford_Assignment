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
  num_train,D,num_class = X.shape[0],X.shape[1], W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.exp(np.dot(X,W))
  for i in range(num_train):
     loss -= np.log(scores[i,y[i]]/np.sum(scores[i,:]))
     dW += np.dot(X[i,:].reshape(D,1), scores[i, :].reshape(1,num_class))/np.sum(scores[i,:])
     dW[:,y[i]] -= X[i,:]
  
  loss /=num_train
  loss += reg*np.sum(W*W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train,D,num_class = X.shape[0],X.shape[1], W.shape[1]
  scores_pre = np.dot(X,W)
  #normalization trick to avoid overflow
  log_c = np.max(scores_pre,axis=1,keepdims=True)
  scores_pre -= log_c
  scores = np.exp(scores_pre)
  
  loss -= np.sum(np.log(scores[np.arange(num_train),y] / np.sum(scores,axis=1)))
  loss /=num_train
  loss  += reg* np.sum(W*W)
  
  temp = np.zeros(scores.shape)
  temp[np.arange(num_train), y] =1
  dW -=  np.dot(X.T , temp-scores/np.sum(scores,axis=1,keepdims=True)) 
  dW /= num_train
  dW += 2*reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

