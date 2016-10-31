#__author__ = 'linkaixiang'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import cPickle as pickle



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

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


def dict_to_nparray(grad_dictionary, n_layers):
    """ Convert dictionary to a numpy array checked

    :param grad_dictionary:
    :param n_nodes:
    :return:
    """

    x = []
    for ii in range(n_layers):
        key = "w" + str(ii + 1)
        temp_v = grad_dictionary[key].flatten()
        x = np.concatenate((x, temp_v), axis=0)

    return x

def nparray_to_dictionary(x, n_feat, n_nodes, n_layers):
    """ Convert a numpy array to a dictionary, checked

    :param x:
    :param n_feat: input features of network, first layer hidden nodes
    :param n_nodes: number of nodes
    :param n_layers: excluding first layers
    :return:
    """
    start_point = 0
    endpoint = 0
    grad_dictionary = {}

    for ii in range(n_layers):
        if ii == 0:
            endpoint += n_feat * n_nodes[0]
        else:
            endpoint += n_nodes[ii] * n_nodes[ii-1]

        xtemp = x[start_point:endpoint]

        if ii == 0:
            grad_dictionary["w" + str(ii + 1)] = xtemp.reshape(n_nodes[0], n_feat)
        else:
            grad_dictionary["w" + str(ii + 1)] = xtemp.reshape(n_nodes[ii], n_nodes[ii-1])

        start_point = endpoint

    return grad_dictionary

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

