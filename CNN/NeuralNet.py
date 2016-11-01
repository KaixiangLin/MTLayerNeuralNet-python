# __author__ = 'linkaixiang'

import numpy as np
import nn_utilities as nnu
import math
class NeuralNet:

    def __init__(self, n_layer, n_nodes,n_feat, func_num):
        ''' @:param: n_layer: number of layers
            @:param: n_nodes: list of numbers e.g. [1,2,3], if it's 3 layers and 1,2,3 hidden nodes respectively
            @:param: n_feat: the number of input features
            @:param: w start from 1
            @:param: h start from 1, h_1 is input value
        '''

        self.train_acc = 0
        self.valid_acc = 0

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
            self.model["w"+str(i+2)] = np.random.randn(n_nodes[i+1], n_nodes[i]) / np.sqrt(n_nodes[i]) # initialization

        self.hidden["h1"] = np.zeros((self.n_feat, 1))
        for i in range(n_layer):
            self.hidden["h"+str(i+2)] = np.zeros((n_nodes[i], 1))

        self.grad_model = {k: np.zeros_like(v) for k, v in self.model.iteritems()}

    def activation(self, y):
        ''' @:param w:   m x n numpy array
            @:param x:   n x 1 numpy array
        '''

        # y = np.dot(w, x)
        h = nnu.activation_funcs(self.func_num, y)
        return h


    def feedforward(self, x):
        ''' get the output value of multi-neural network

        :param x: input data
        :return:
        '''

        self.hidden["h1"] = x  # output of activation function
        self.z["h1"] = x  # input of activation function
        for i in range(2, self.n_layer+1):
            self.z["h" + str(i)] = np.dot(self.model["w" + str(i-1)], self.hidden["h" + str(i-1)])
            self.hidden["h" + str(i)] = self.activation(self.z["h" + str(i)])

        self.z["h"+str(self.n_layer+1)] = np.dot(self.model["w" + str(self.n_layer)], self.hidden["h" + str(self.n_layer)])
        self.hidden["h" + str(self.n_layer + 1)] = nnu.activate_softmax(self.z["h"+str(self.n_layer+1)])

        return self.hidden["h"+str(self.n_layer+1)]  # return last hidden layer


    def backprop(self,err):
        ''' back propagation
        :return: the gradient with respect to the error.
        '''

        delta = {}
        # last layer gradient delta (error)

        delta[str(self.n_layer+1)] = err # delta_4

        for ii in range(1, self.n_layer+1)[::-1]:

            delta_w = np.dot(delta[str(ii+1)], self.model["w"+str(ii)])  # ii  = 2
            delta[str(ii)] = np.multiply(delta_w, nnu.grad_activation(self.func_num, self.z["h" + str(ii)]))   # delta_3 3x1
            self.grad_model["w"+str(ii)] = np.outer(delta[str(ii+1)], self.hidden["h"+str(ii)].T)

        return self.grad_model

    def loss_function_crossentropy(self, y_pred, y_true):
        """ -  y_true * log y_pred
        :param y_pred: predicted class distribution, 10x1 vector
        :param y_true: true class distribution, 10x1 vector
        :return:
        """

        loss_func = - np.sum(np.multiply(y_true, np.log(y_pred)))

        return loss_func

    def loss_function_crossentropy_grad(self, y_pred, y_true):
        """
        :param y_pred:
        :param y_true:
        :return:
        """

        err = y_pred - y_true  # d L / d z_i   for i = 1:class
        return err


    def loss_function_l2(self, y_pred, y_true):
        '''

        :param y_pred: the prediction of neural network
        :param y_ture: the true label
        :return:
        '''

        loss_func = np.linalg.norm(y_pred-y_true)

        return 0.5 * loss_func ** 2

    def loss_function_l2_grad(self, y_pred, y_true):
        """ return the gradient of loss function with respect to the y_pred

        :param y_pred:
        :param y_true:
        :return:
        """
        err = y_pred - y_true

        return err


