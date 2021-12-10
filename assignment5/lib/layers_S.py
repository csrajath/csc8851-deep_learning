from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    out = x.reshape(N, -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N, -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = dout
    dx[x <= 0] = 0
    return dx


def conv_forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = (H + 2 * pad - HH) / stride + 1
      W' = (W + 2 * pad - WW) / stride + 1
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + int((H + 2 * pad - HH) / stride)
    W_out = 1 + int((W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i*stride:i *
                                 stride+HH, j*stride:j*stride+WW]
            for k in range(F):
                out[:, k, i, j] = np.sum(
                    x_pad_masked * w[k, :, :, :], axis=(1, 2, 3))

    out = out + (b)[None, :, None, None]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache

    pad = conv_param['pad']
    stride = conv_param['stride']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_filter = dout.shape[2]
    W_filter = dout.shape[3]

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    for i in range(N):
        for z in range(F):
            for j in range(H_filter):
                h_start = j * stride
                for k in range(W_filter):
                    w_start = k * stride
                    dx[i, :, h_start:(h_start + HH), w_start:(w_start + WW)
                       ] += w[z, :, :, :] * dout[i, z, j, k]
                    dw[z, :, :, :] += x[i, :,
                                        h_start:(h_start + HH), w_start:(w_start + WW)] * dout[i, z, j, k]

    db = dout.sum(axis=(0, 2, 3))

    return dx[:, :, pad:-pad, pad:-pad], dw, db


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = int((H-HH)/stride)+1
    W_out = int((W-WW)/stride)+1
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i*stride: i*stride+HH, j*stride: j*stride+WW]
            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    x, pool_param = cache

    N, C, H, W = x.shape
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    stride = pool_param['stride']

    H_filter = (H - pool_H) / stride + 1
    W_filter = (W - pool_W) / stride + 1

    dx = np.zeros_like(x)

    for i in range(N):
        for z in range(C):
            for j in range(H_filter):
                for k in range(W_filter):
                    dpatch = np.zeros((pool_H, pool_W))
                    input_patch = x[i, z, j *
                                    stride:(j * stride + pool_H), k * stride:(k * stride + pool_W)]
                    idxs_max = np.where(input_patch == input_patch.max())
                    dpatch[idxs_max[0], idxs_max[1]] = dout[i, z, j, k]
                    dx[i, z, j * stride:(j * stride + pool_H),
                       k * stride:(k * stride + pool_W)] += dpatch

    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
