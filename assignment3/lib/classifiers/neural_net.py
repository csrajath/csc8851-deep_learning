from __future__ import print_function
import numpy as np
from past.builtins import xrange


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C). ReLU instead of Sigmoid should be used.                     #
        #############################################################################
        i_scores = np.maximum(0, (np.dot(X, W1) + b1))
        scores = np.dot(i_scores, W2)+b2
       #############################################################################
       #                              END OF YOUR CODE                             #
       #############################################################################

       # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        normalization = scores - (np.max(scores, axis=1, keepdims=True))
        p_reduce = np.exp(normalization) / \
            (np.add.reduce(np.exp(normalization), axis=1, keepdims=True))
        loss = np.sum(-np.log(p_reduce[np.arange(N), y]))
        loss = (loss/N) + (reg * (np.sum(W1 * W1) + np.sum(W2 * W2)))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        der1 = p_reduce.copy()
        der1[range(N), list(y)] -= 1
        der1 /= N
        grads['W2'] = i_scores.T.dot(der1) + reg * W2
        grads['b2'] = np.sum(der1, axis=0)

        grads['W1'] = X.T.dot(((i_scores > 0) * (der1.dot(W2.T)))) + reg * W1
        grads['b1'] = np.sum(((i_scores > 0) * (der1.dot(W2.T))), axis=0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            ith = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[ith]
            y_batch = y[ith]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['W2'] += - learning_rate * grads['W2']
            self.params['b2'] += - learning_rate * grads['b2']
            self.params['W1'] += - learning_rate * grads['W1']
            self.params['b1'] += - learning_rate * grads['b1']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        i_scores = np.maximum(
            0, X.dot(self.params['W1']) + self.params['b1'])
        y_pred = np.argmax(
            (i_scores.dot(self.params['W2']) + self.params['b2']), axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
