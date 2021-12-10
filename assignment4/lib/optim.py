import numpy as np


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    momentum = config['momentum']*v
    l_rate = config['learning_rate'] * dw
    v = momentum - l_rate
    next_w = w + v

    config['velocity'] = v

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def rmsprop(x, dx, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x #
    # in the next_x variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    decayRate = (1 - config['decay_rate'])
    config['cache'] = config['decay_rate'] * \
        config['cache'] + decayRate * (dx**2)
    cache_sq = np.sqrt(config['cache'])
    d_next_x = (cache_sq + config['epsilon'])
    next_x = (x - config['learning_rate'] * dx) / d_next_x

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config


def adam(x, dx, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in #
    # the next_x variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    ###########################################################################
    config['t'] += 1
    beta1_diff = (1 - config['beta1'])
    config['m'] = config['beta1'] * config['m'] + beta1_diff * dx
    config['v'] = config['beta2'] * config['v'] + \
        (1 - config['beta2']) * (dx**2)
    beta_m = config['m'] / (1 - config['beta1']**config['t'])
    vb = config['v'] / (1 - config['beta2']**config['t'])
    next_x = x - config['learning_rate'] * \
        beta_m / (np.sqrt(vb) + config['epsilon'])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config
