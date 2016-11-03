import nn_train as nnt
import nn_utilities as nnu
import numpy as np
from scipy import signal
from configure import FLAGS
from ConvoNN import NeuralNet
from nn_train import StochasticGradientDescent

# a = np.array([[1,2,3], [3,2,1], [3,6,1]])
# f = np.array([[1, -1], [-2, 1]])
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
'''test repeat '''
# a = np.array([[1, 2], [1, 2]])
# print(a.shape)
# b = np.repeat(a[:, :, np.newaxis], 3, axis=2)
#
# print(b.shape), b[:,:,0], b[:,:,1], b[:,:,2]

# c = a.reshape(9,1)
# b = c.reshape(3,3)
# print a, c, b

'''max pooling'''
# aa = np.random.randn(4, 4, 2)
# bb, cc =  nnu.max_pooling(aa, 2)
# dd = aa * cc
# C = 1
#
# inverse_aa = nnu.max_pool_inv(cc, bb, 2)
# print aa[:,:,C]
# print inverse_aa[:,:,C]
# print bb[:,:,C]
# print cc[:,:,C]
# print dd[:,:,C]



# '''test conv'''
# aa = np.random.rand(4, 4, 2)
# bb = np.random.rand(2, 2, 3)
# cc = nnu.convolution_tensor_tensor(aa, bb)
# print cc.shape

# aa = np.random.rand(4, 4, 3)
# bb = np.random.rand(2, 2, 3)
# cc = nnu.convolution_backprop(aa, bb)
# print cc.shape

'''test forward'''
# features = np.random.randn(28, 28)
# nn_model = NeuralNet(28, 3)
# print nn_model.feedforward(features)


'''test backward'''
# # # aa = np.random.randn(4, 4, 2)
# # # print np.sum(aa, axis=2), np.sum(aa[0,0, :])
# features = np.random.randn(28, 28)
# err = np.random.randn(10, 1)
# nn_model = NeuralNet(28, 3, FLAGS.filter1_channel, FLAGS.filter2_channel,FLAGS.n_node_linear1_row)
# nn_model.feedforward(features)
# nn_model.backprop(err)



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


# SGD
# trainaccs = [0.3426, 0.4405, 0.4779, 0.5213,0.5428, 0.5549]
# validaccs = [0.3315, 0.4448, 0.4767, 0.5344, 0.5429, 0.5549]
# figname = "../results/" + "sgd_cnn.png"


# adamdelta
# trainaccs = [0.3505, 0.4817, 0.5303, 0.573,0.6008, 0.6309, 0.6506, 0.6839, 0.6815]
# validaccs = [0.351, 0.4759, 0.5276, 0.5747,0.5977, 0.6274, 0.6428, 0.671, 0.6829]
# figname = "../results/" + "adamdelta_cnn.png"

# Nesterov
# trainaccs = [0.3263, 0.3049, 0.2829, 0.3984]
# validaccs = [0.3303, 0.3061, 0.289, 0.4085]
# figname = "../results/" + "Nesterov_cnn.png"

# adamgrad
trainaccs = [0.2665, 0.318, 0.3912, 0.4186, 0.4444]
validaccs = [0.2732, 0.3283, 0.3919, 0.4158, 0.4358]
figname = "../results/" + "adamgrad_cnn.png"
nnu.plot_list_acc(trainaccs, validaccs, figname)


