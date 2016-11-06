#__author__ = 'linkaixiang'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import cPickle as pickle
from scipy import signal

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


def activate_softmax(x):
    """ e^x_i / \sum_k e^x_k
    :param x: 10 by 1 vector np array
    :return: y: 10 by 1 vector np array
    """
    C = np.exp(-np.max(x))
    x_new = x + np.log(C)
    exp_xnew = np.exp(x_new)
    y = exp_xnew / np.sum(exp_xnew)
    return y


def activation_funcs(func_num, x):
    """ @:param func_num: choose different type of activate functions
        func_num = 1: relu
    """
    output = 0

    if func_num == 1:
        # output = np.copy(x)
        output = np.maximum(x, 0)
    elif func_num == 2:
        output = sigmoid(x)
    elif func_num == 3:
        output = tanh(x)
    else:
        print("ERROR WRONG Activation function")
        pass

    return output

def grad_activation(func_num, x):

    gradient = 0
    if func_num == 1:
        gradient = np.copy(x)
        gradient[x<0] = 0
        gradient[x>=0] = 1

    elif func_num == 2:
        gradient = np.multiply(sigmoid(x), 1 - sigmoid(x))

    elif func_num == 3:
        gradient = 1 - tanh(x) ** 2
    else:
        print("ERROR WRONG GRADIENT")

    return gradient

def conv(image, filter):
    """
    :param image: 2 dimensional np array NxN
    :param filter: 2 dimensional np array MxM
    :return: (N-M+1)x(N-M+1)
    """
    h = signal.convolve(image, filter, 'valid')
    return h

def convolution_tensor(image, filtertensor):
    """
    :param image:  2 dimensional np array NxN
    :param filtertensor:  2 dimensional np array MxMx channels
    :return:
    """
    M, _, C = filtertensor.shape
    N, _ = image.shape

    L = N - M + 1
    H = np.zeros((L, L, C))
    for ii in range(C):
        H[:, :, ii] = conv(image, np.rot90(filtertensor[:, :, ii],2))

    return H

def convolution_tensor_tensor(imagetensor, filtertensor):
    """ conv 2 tensors.  use MxM,C2 conv all C1 slices(NxN) in imagetensor[:,:,c] and add C1 all
     to get a tensor with LxL, C2, where L = N-M + 1, valid conv

    :param imagetensor: NxN, C1
    :param filtertensor: MxM, C2
    :return:
    """
    N, _, C1 = imagetensor.shape
    M, _, C2 = filtertensor.shape

    L = N - M + 1
    # tensor_new = np.zeros((L, L, C2))
    # for cc in range(C1):
    #     tensor_new += convolution_tensor(imagetensor[:, :, cc], filtertensor)

    image_new = np.sum(imagetensor, axis=2)
    tensor_new = convolution_tensor(image_new, filtertensor)
    return tensor_new

def max_pool_inv(tensor_pos, tensor_pooled, m):
    """ inverse process of max pooling , err pro back
    :param tensor_pos: 8x8, 64
    :param tensor_pooled: 4x4, 64
    :return:
    """
    tensor_new = np.zeros((tensor_pos.shape))
    N, _, C = tensor_pos.shape

    for a in range(N):
        for b in range(N):
            for ii in range(C):
                if tensor_pos[a, b, ii] == 1:
                    tensor_new[a, b, ii] = tensor_pooled[a/m, b/m, ii]
    return tensor_new


def max_pooling(tensor, m):
    """ pooling each NxN slice of tensor into N/m by N/m slice

    :param tensor: N by N by C tensor
    :param m: m x m pooling
    :return:
    """
    N, _, C = tensor.shape
    r = N/m
    tensor_new = np.zeros((r, r, C))
    tensor_pos = np.zeros((N, N, C)) # remember the position of pooling, using mask, 1 denote from this point
    for a in range(r):
        for b in range(r):
            for ii in range(C):
                tensor_new[a, b, ii] = np.amax(tensor[a*m:(a+1)*m, b*m:(b+1)*m, ii])
                pos = np.argmax(tensor[a*m:(a+1)*m, b*m:(b+1)*m, ii])
                aa = pos / m
                bb = pos - aa*m
                tensor_pos[a*m + aa, b*m + bb, ii] = 1
    return tensor_new, tensor_pos

def convolution_backprop(error_tensor, filter):
    """ bp the error back, the channel of error tensor and filter should be same
    :param errr_tensor: 8x8, 64
    :param filter: 5x5, 64
    :return:
    """
    N, _, C1 = error_tensor.shape
    M, _, C2 = filter.shape
    if C1 != C2:
        print("convolution_backprop error")
    L = N + M -1
    tensor_new = np.zeros((L, L))
    for ii in range(C1):
        tensor_new += signal.convolve(error_tensor[:, :, ii], filter[:, :, ii], 'full')

    return tensor_new

def convolution_gradient(delta, image):
    """
    :param delta: C dimensional np array rxr, error,
    :param image: 2 dimensional np array NxN, hidden
    :return:
    """
    N1, N2 = image.shape # N1 == N2
    r1, r2, C = delta.shape # r1 == r2
    M = N1 + 1 - r1 # filter size

    grad_filter = np.zeros((M, M, C))
    for cc in range(C):
        for a in range(M):
            for b in range(M):
                grad_filter[a, b, cc] += np.sum(image[a:a+r1, b:b+r1] * delta[:, :, cc])
    return grad_filter

def reshape_vec_tensor(f_vector, m, C):
    """ reshape a vector into a tensor m x m x C with C channel(C filters), mxm size of filter

    :param f_vector:  (mxmxC) by 1 vector
    :param m:
    :param C:
    :return:
    """
    tensor = np.zeros((m,m, C))
    fsize = m * m
    for ii in range(C):
        tensor[:, :, ii] = f_vector[ii*fsize:(ii+1)*fsize].reshape(m, m)

    return tensor

def dict_to_nparray(grad_dictionary):
    """ Convert dictionary to a numpy array checked

    :param grad_dictionary:
    :param n_nodes:
    :return:
    """

    keys = list(grad_dictionary.keys())
    n_layers = len(keys)
    x = []
    for ii in range(n_layers):
        key = "w" + str(ii + 1)
        temp_v = grad_dictionary[key].flatten()
        x = np.concatenate((x, temp_v), axis=0)

    return x

def nparray_to_dictionary(x, new_dictionary):
    """ Convert a numpy array to a dictionary, same structure as new_dictionary

    :param x:
    :param n_feat: input features of network, first layer hidden nodes
    :param n_nodes: number of nodes
    :param n_layers: excluding first layers
    :return:
    """
    start_point = 0
    endpoint = 0
    grad_dictionary = {}
    keys = list(new_dictionary.keys())
    layers = len(keys)

    for ii in range(layers):
        k = "w" + str(ii+1)
        v = new_dictionary[k]
        endpoint = start_point + v.size
        grad_dictionary[k] = x[start_point:endpoint].reshape(v.shape)
        start_point = endpoint


    return grad_dictionary

# def nparray_to_dictionary(x, n_feat, n_nodes, n_layers):
#     """ Convert a numpy array to a dictionary, checked
#
#     :param x:
#     :param n_feat: input features of network, first layer hidden nodes
#     :param n_nodes: number of nodes
#     :param n_layers: excluding first layers
#     :return:
#     """
#     start_point = 0
#     endpoint = 0
#     grad_dictionary = {}
#
#     for ii in range(n_layers):
#         if ii == 0:
#             endpoint += n_feat * n_nodes[0]
#         else:
#             endpoint += n_nodes[ii] * n_nodes[ii-1]
#
#         xtemp = x[start_point:endpoint]
#
#         if ii == 0:
#             grad_dictionary["w" + str(ii + 1)] = xtemp.reshape(n_nodes[0], n_feat)
#         else:
#             grad_dictionary["w" + str(ii + 1)] = xtemp.reshape(n_nodes[ii], n_nodes[ii-1])
#
#         start_point = endpoint
#
#     return grad_dictionary

def dict_add(d1, d2):
    """ add two dictionary, d_new[key] = d1[key] + d2[key]

    :param d1:
    :param d2:
    :return
    """
    d_new = {}
    for k, v in d1.iteritems():
        d_new[k] = v + d2[k]

    return d_new

def dict_minus(d1, d2):
    """ minus two dictionary, d_new[key] = d1[key] - d2[key]

    :param d1:
    :param d2:
    :return
    """
    d_new = {}
    for k, v in d1.iteritems():
        d_new[k] = v - d2[k]

    return d_new


def dict_mulscala(d1, a):
    """ multiply scala

    :param d1: dictionary
    :param a: scala
    :return:
    """
    d_new = {}
    for k, v in d1.iteritems():
        d_new[k] = v * a
    return d_new

def dict_mul(d1, d2):
    """ element wise multiply two dictionary

    :param d1: dictionary
    :param a: scala
    :return:
    """
    d_new = {}
    for k, v in d1.iteritems():
        d_new[k] = np.multiply(v, d2[k])
    return d_new


def batch_data(x, y, batch_size):
    '''Input numpy array, and batch size'''
    data_size = len(x)
    index = np.random.permutation(data_size)
    # index = range(data_size)
    batch_index = index[:batch_size]

    batch_x = x[batch_index]
    batch_y = y[batch_index]
    return batch_x, batch_y

def save_model(nn_model, filename):
    """ Save model to the file

    :param nn_model:
    :param filename:
    :return:
    """

    with open(filename, "wb") as f:
        pickle.dump(nn_model, f, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    """ Load network model from file

    :param filename:
    :return:
    """
    with open(filename, "rb") as f:
        nn_model = pickle.load(f)

    return nn_model


def plot_list(loss_val, figname, closefig):
    fig = plt.figure(1)
    plt.plot(loss_val)
    plt.savefig(figname)
    if closefig == 1:
        plt.close(fig)

def plot_list_acc(train, valid, figname):
    fig = plt.figure(1)
    plt.plot(train, 'g--', label='train')
    plt.plot(valid, 'g^', label='valid')
    plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=2,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(figname)
    plt.close(fig)

