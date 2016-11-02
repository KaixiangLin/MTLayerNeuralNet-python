import nn_train as nnt
import nn_utilities as nnu
import numpy as np
from scipy import signal
from configure import FLAGS
from ConvoNN import NeuralNet
from nn_train import StochasticGradientDescent

a = np.array([[1,2,3], [3,2,1], [3,6,1]])
f = np.array([[1, -1], [-2, 1]])
# print a
# print nnu.tanh(a)
# print nnu.tanh(1)

# testing convol
# print a
# print f
# d = signal.convolve(a, f, 'valid')   # filp the filter and them element wise multiply
#
# print d
#
# a = np.zeros((3,2,2))
# c, _, e = a.shape
#
# a[:,:,1] = np.random.randn(3,2)
#
# print a.reshape(12,1)

c = a.reshape(9,1)
b = c.reshape(3,3)
print a, c, b



# # testing
# FLAGS.n_layer = len(FLAGS.n_nodes)
# features = np.random.randn(28, 28)
# labels = np.random.randn(1, 2)
#
# n_layer = 2  # the last layer is the output of network
#
# nn_model = NeuralNet(28, 2)
#
# nn_model.feedforward(features)
# # datatuple = tuple([features, labels, features, labels, features, labels])
# # StochasticGradientDescent(datatuple, nn_model)
# err = np.random.randn(1, 10)
# nn_model.backprop(err)