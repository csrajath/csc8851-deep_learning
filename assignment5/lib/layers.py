from builtins import range
import numpy as np


def affine_forward(x, w, b):
    N = x.shape[0]
    out = x.reshape(N, -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(N, -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout
    dx[x <= 0] = 0
    return dx


def conv_forward(x, w, b, conv_param):
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    x_new = np.pad(
        x, ((0, 0), (0, 0), (conv_param['pad'], conv_param['pad']), (conv_param['pad'], conv_param['pad'])), 'constant')
    (N, C, H, W), (F, _, HH, WW) = x.shape, w.shape
    out = np.zeros((N, F, (int(
        1 + (H + 2 * conv_param['pad'] - HH) / (conv_param['stride']))), (int(
            1 + (W + 2 * conv_param['pad'] - WW) / (conv_param['stride'])))))

    for i in range((int(
            1 + (H + 2 * conv_param['pad'] - HH) / (conv_param['stride'])))):
        for j in range((int(
                1 + (W + 2 * conv_param['pad'] - WW) / (conv_param['stride'])))):
            masked_values = x_new[:, :, i*(conv_param['stride']):i *
                                  (conv_param['stride'])+HH, j*(conv_param['stride']):j*(conv_param['stride'])+WW]
            for c in range(F):
                out[:, c, i, j] = np.sum(
                    masked_values * w[c, :, :, :], axis=(1, 2, 3)) + b[c]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):

    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))
    dx_pad = np.zeros_like(x_pad)

    for i in range(H_new):
        for j in range(W_new):
            x_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for c in range(F):
                dw[c, :, :, :] += np.sum(x_masked * (dout[:, c, i, j])
                                         [:, None, None, None], axis=0)
            for n in range(N):
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum(
                    w[:, :, :, :]*(dout[n, :, i, j])[:, None, None, None], axis=(0))

    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db


def max_pool_forward(x, pool_param):
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    out = np.zeros((N, C, (int(
        1 + (H - (pool_param['pool_height'])) / (pool_param['stride']))), (int(1 + (W - (pool_param['pool_width'])) / (pool_param['stride'])))))

    for i in range((int(
            1 + (H - (pool_param['pool_height'])) / (pool_param['stride'])))):
        for j in range((int(1 + (W - (pool_param['pool_width'])) / (pool_param['stride'])))):
            masked_values = x[:, :, i*(pool_param['stride']):i*(pool_param['stride']) +
                              (pool_param['pool_height']), j*(pool_param['stride']):j*(pool_param['stride'])+(pool_param['pool_width'])]
            out[:, :, i, j] = np.max(masked_values, axis=(2, 3))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    x, pool_param = cache
    dx = np.zeros_like(x)
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)

    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i*stride:i*stride +
                         pool_height, j*stride:j*stride+pool_width]
            max_mask = np.max(x_masked, axis=(2, 3))
            temp_mask = x_masked == (max_mask)[:, :, None, None]
            dx[:, :, i*stride:i*stride+pool_height, j*stride:j*stride +
                pool_width] += temp_mask * (dout[:, :, i, j])[:, :, None, None]

    return dx


def softmax_loss(x, y):
    N = x.shape[0]
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
