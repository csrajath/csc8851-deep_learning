
# coding: utf-8

# # Convolutional Networks
# In this assignment we will implement convolutional and maxpooling layers that are used in convolutional networks.

# In[ ]:

# As usual, a bit of setup
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from lib.data_utils import get_CIFAR10_data
from lib.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from lib.layers import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# In[ ]:

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)


# # Convolution: Forward pass
# The core of a convolutional network is the convolution operation. In the file `lib/layers.py`, implement the forward pass for the convolution layer in the function `conv_forward`. 
# 
# You don't have to worry too much about efficiency; just write the code in whatever way you find most clear.
# 
# You can test your implementation by running the following:

# In[ ]:

x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out, _ = conv_forward(x, w, b, conv_param)
correct_out = np.array([[[[-0.08759809, -0.10987781],
                           [-0.18387192, -0.2109216 ]],
                          [[ 0.21027089,  0.21661097],
                           [ 0.22847626,  0.23004637]],
                          [[ 0.50813986,  0.54309974],
                           [ 0.64082444,  0.67101435]]],
                         [[[-0.98053589, -1.03143541],
                           [-1.19128892, -1.24695841]],
                          [[ 0.69108355,  0.66880383],
                           [ 0.59480972,  0.56776003]],
                          [[ 2.36270298,  2.36904306],
                           [ 2.38090835,  2.38247847]]]])

# Compare your output to ours; difference should be around 2e-8
print('Testing conv_forward')
print('difference: ', rel_error(out, correct_out))


# # Convolution: Backward pass
# We have provided the implementation of the backward pass for the convolution operation in the function `conv_backward` in the file `lib/layers.py`.
# Run the following to check your backward pass with a numeric gradient check. If your forward pass is correct, the relative errors should be very small (lower than 1e-6).

# In[ ]:

np.random.seed(231)
x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(lambda x: conv_forward(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward(x, w, b, conv_param)[0], b, dout)

out, cache = conv_forward(x, w, b, conv_param)
dx, dw, db = conv_backward(dout, cache)

# Your errors should be around 1e-8'
print('Testing conv_backward function')
print('dx error: ', rel_error(dx, dx_num))
print('dw error: ', rel_error(dw, dw_num))
print('db error: ', rel_error(db, db_num))


# # Max pooling: Forward
# Implement the forward pass for the max-pooling operation in the function `max_pool_forward` in the file `lib/layers.py`. Again, don't worry too much about computational efficiency.
# 
# Check your implementation by running the following:

# In[ ]:

x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = max_pool_forward(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

# Compare your output with ours. Difference should be around 1e-8.
print('Testing max_pool_forward function:')
print('difference: ', rel_error(out, correct_out))


# # Max pooling: Backward
# The implementation of the backward pass for the max-pooling operation is also provided in the function `max_pool_backward` in the file `lib/layers.py`.
# 
# Verify the implementation with numeric gradient checking by running the following:

# In[ ]:

np.random.seed(231)
x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 2, 4, 4)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward(x, pool_param)[0], x, dout)

out, cache = max_pool_forward(x, pool_param)
dx = max_pool_backward(dout, cache)

# Your error should be around 1e-12
print('Testing max_pool_backward function:')
print('dx error: ', rel_error(dx, dx_num))
