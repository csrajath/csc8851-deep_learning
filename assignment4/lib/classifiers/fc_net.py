from builtins import range
from builtins import object
import numpy as np

from lib.layers import *
from lib.layer_utils import *


class TwoLayerNet(object):

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):

        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        randDims = np.random.randn(input_dim, hidden_dim)
        randDims_i = np.random.randn(hidden_dim, num_classes)
        self.params['W1'] = weight_scale * randDims
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * randDims_i
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        reg = self.reg

        hScore, hCache = affine_relu_forward(X, W1, b1)
        scores, oCache = affine_forward(hScore, W2, b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        ############################################################################

        loss, softmax_grad = softmax_loss(scores, y)
        sumWeight1, sumWeight2 = np.sum(W1**2), np.sum(W2**2)
        sumOfSums = sumWeight1 + sumWeight2
        loss += (reg * (sumOfSums))

        hidden_grad, grads['W2'], grads['b2'] = affine_backward(
            softmax_grad, oCache)

        grads['W2'] += reg*W2

        input_grad, grads['W1'], grads['b1'] = affine_relu_backward(
            hidden_grad, hCache)

        grads['W1'] += reg*W1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, dtype=np.float32):

        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        ############################################################################
        # net_dims = [input_dim] + hidden_dims + [num_classes]

        self.params['W1'], self.params['b1'] = {}, {}

        self.params['W'+str(1)] = weight_scale * \
            np.random.randn(input_dim, hidden_dims[0])
        self.params['b'+str(1)] = np.zeros(hidden_dims[0])

        for i in range(1, self.num_layers-1):
            self.params['W'+str(i+1)] = weight_scale * \
                np.random.randn(hidden_dims[i-1], hidden_dims[i])
            self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])

        self.params['W'+str(self.num_layers)] = weight_scale * \
            np.random.randn(hidden_dims[-1], num_classes)
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        ############################################################################
        scores = {}
        scores[0] = X
        temp = {}

        for i in range(1, self.num_layers+1):
            Ws = self.params['W'+str(i)]
            Bs = self.params['b'+str(i)]
            scores[i], temp[i] = affine_relu_forward(scores[i-1], Ws, Bs)
            if i == self.num_layers:
                scores[i], temp[i] = affine_forward(scores[i-1], Ws, Bs)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores[self.num_layers]

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        xGrads = {}
        W_reg_sum = 0
        reg = self.reg

        loss, softmax_grad = softmax_loss(scores[self.num_layers], y)

        for i in range(1, self.num_layers+1):
            W_reg_sum += np.sum(self.params['W'+str(i)])

        loss = loss + reg * W_reg_sum

        xGrads[self.num_layers], grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] \
            = affine_backward(softmax_grad, temp[self.num_layers])
        for i in range(self.num_layers-1, 0, -1):
            xGrads[i], grads['W'+str(i)], grads['b'+str(i)
                                                ] = affine_relu_backward(xGrads[i+1], temp[i])
            grads['W'+str(i)] = grads['W'+str(i)] + reg*self.params['W'+str(i)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
