# __author__ = 'linkaixiang'

import numpy as np
import nn_utilities as nnu
import math
class NeuralNet:

    def __init__(self, n_layer, n_nodes,n_feat, func_num=2):
        ''' @:param: n_layer: number of layers
            @:param: n_nodes: list of numbers e.g. [1,2,3], if it's 3 layers and 1,2,3 hidden nodes respectively
            @:param: n_feat: the number of input features
            @:param: w start from 1
            @:param: h start from 0, h_0 is input value
        '''
        self.n_layer = n_layer
        self.n_nodes = n_nodes
        self.n_feat = n_feat
        self.func_num = func_num
        self.model = {}
        self.hidden = {}
        self.z = {}  # input to the activation function
        self.grad_model = {}

        #  todo didn't consider batch size, culmulate the batch size
        self.model["w1"] = np.random.randn(n_nodes[0], n_feat) / np.sqrt(math.sqrt(float(n_feat)))
        for i in range(n_layer-1):
            self.model["w"+str(i+2)] = np.random.randn(n_nodes[i+1],n_nodes[i]) / np.sqrt(n_nodes[i]) # initialization

        for i in range(n_layer):
            self.hidden["h"+str(i+1)] = np.zeros((n_nodes[i],1))

        self.grad_model = {k: np.zeros_like(v) for k, v in self.model.iteritems()}

    def activation(self, y):
        ''' @:param w:   m x n numpy array
            @:param x:   n x 1 numpy array
        '''

        # y = np.dot(w, x)
        h = nnu.activation_funcs(self.func_num, y)
        return h



    def loss_function_l2(self, y_pred, y_true):
        '''

        :param y_pred: the prediction of neural network
        :param y_ture: the true label
        :return:
        '''

        loss_func = np.linalg.norm(y_pred-y_true)

        return loss_func ** 2


    def feedforward(self, x):
        ''' get the output value of multi-neural network

        :param x: input data
        :return:
        '''

        self.hidden["h0"] = x
        for i in range(self.n_layer-1):
            i += 1
            self.z["h" + str(i)] = np.dot(self.model["w"+ str(i)], self.hidden["h" + str(i-1)])
            self.hidden["h" + str(i)] = self.activation(self.z["h" + str(i)])

        self.hidden["h"+str(self.n_layer)] = np.dot(self.model["w"+ str(self.n_layer)], self.hidden["h" + str(self.n_layer-1)])

        return self.hidden["h"+str(self.n_layer)]  # return last hidden layer


    def backprop(self,err):
        ''' back propagation
        :return:
        '''

        # last layer gradient
        self.grad_model["w" + str(self.n_layer)] = np.multiply(err, self.hidden[self.n_layer - 1])

        delta = {}
        delta[str(self.n_layer+1)] = err # delta_4

        for ii in range(1, self.n_layer+1)[::-1]:
            delta_w = np.dot(delta[str(ii+1)], self.model[str(ii)])  # ii  = 2
            delta[str(ii)] = np.multiply(delta_w, nnu.grad_activation(self.func_num, self.z["h" + str(ii)])) # delta_3 3x1
            self.grad_model["w"+str(ii)] = np.outer(delta[str(ii+1)], self.hidden["h"+str(ii)].T)

        return self.grad_model








