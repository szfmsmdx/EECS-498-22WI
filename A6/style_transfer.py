"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a6_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    # Replace "pass" statement with your code
    loss = content_weight * torch.sum((content_current - content_original)**2)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    # Replace "pass" statement with your code
    N,C,H,W = features.shape
    features = features.view(N,C,-1)
    gram = torch.bmm(features,features.transpose(1,2))
    
    if normalize:
       gram /= (H*W*C)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    # Replace "pass" statement with your code
    loss = 0
    for i in range(len(style_layers)):
       gram = gram_matrix(feats[style_layers[i]])
       loss += style_weights[i] * torch.pow((gram-style_targets[i]),2).sum()
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Replace "pass" statement with your code
    H_loss = torch.sum(torch.pow(img[:,:,1:,:]-img[:,:,0:-1,:],2))
    W_loss = torch.sum(torch.pow(img[:,:,:,1:]-img[:,:,:,0:-1],2))
    loss = tv_weight*(H_loss+W_loss)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # TODO: Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################
  # Replace "pass" statement with your code
  N,R,C,H,W = features.shape
  features = features.view(N,R,C,-1)  #(N,R,C,H*W)
  masks = masks.view(N,R,1,-1)  #(N,R,H*W)
  masked_features = features * masks  #(N,R,C,H*W)
  guided_gram = torch.matmul(masked_features,masked_features.transpose(2,3))
  if normalize:
     guided_gram /= (H*W*C)
  return guided_gram
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # Replace "pass" statement with your code
    loss = 0
    for idx, layer_idx in enumerate(style_layers):
       gram = guided_gram_matrix(feats[layer_idx],content_masks[layer_idx])
       loss += style_weights[idx] * torch.sum((gram - style_targets[idx]) ** 2)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
