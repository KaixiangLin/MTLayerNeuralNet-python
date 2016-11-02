# __author__ = 'linkaixiang'

import numpy as np
import nn_utilities as nnu
import math
class NeuralNet:

    def __init__(self, n_feat, func_num):
        ''' @:param: n_layer: number of layers
            @:param: n_nodes: list of numbers e.g. [1,2,3], if it's 3 layers and 1,2,3 hidden nodes respectively
            @:param: n_feat: the number of input features
            @:param: w start from 1
            @:param: h start from 1, h_1 is input value
        '''

        self.train_acc = 0
        self.valid_acc = 0
        self.n_layer = 2
        self.n_feat = n_feat
        self.func_num = func_num
        self.model = {}
        self.hidden = {}
        self.z = {}  # input to the activation function
        self.grad_model = {}

        self.layer1_filter_size = 5
        self.layer1_channel = 1

        # convolution filter
        self.model["w1"] = np.random.randn(self.layer1_filter_size, self.layer1_filter_size, self.layer1_channel) \
                           / np.sqrt(math.sqrt(float(self.layer1_filter_size)))

        # last layer
        self.model["w2"] = np.random.randn(10, (self.n_feat - self.layer1_filter_size + 1) ** 2)

        self.hidden["h2"] = np.zeros((self.n_feat - self.layer1_filter_size + 1, self.n_feat - self.layer1_filter_size + 1))

        self.grad_model = {k: np.zeros_like(v) for k, v in self.model.iteritems()}

    def activation(self, y):
        ''' @:param w:   m x n numpy array
            @:param x:   n x 1 numpy array
        '''

        # y = np.dot(w, x)
        h = nnu.activation_funcs(self.func_num, y)
        return h


    def feedforward(self, x):
        ''' Define CNN network

        :param x: input data 28 * 28 image
        :return:
        '''

        self.hidden["h1"] = x  # output of activation function
        self.z["h1"] = x  # input of activation function

        # convolution layer
        self.z["h2"] = nnu.convolution_tensor(self.hidden["h1"], self.model["w1"])   # output tensor after convo

        # z1, z2, z3 = self.z["h2"].shape
        self.hidden["h2"] = self.z["h2"].reshape(self.z["h2"].size, 1)

        #soft max layer
        self.z["h3"] = np.dot(self.model["w2"], self.hidden["h2"])
        self.hidden["h3"] = nnu.activate_softmax(self.z["h"+str(self.n_layer+1)])



        return self.hidden["h"+str(self.n_layer+1)]  # return last hidden layer


    def backprop(self, err):
        ''' back propagation
        :return: the gradient with respect to the error.
        '''

        delta = {}
        # last layer gradient delta (error)

        delta[str(self.n_layer+1)] = err # delta_3
        self.grad_model["w2"] = np.outer(delta["3"], self.hidden["h2"].T)

        delta_w = np.dot(self.model["w2"].T, delta["3"])  # ii  = 2
        delta["2"] = delta_w.reshape(self.z["h2"][:, :, 0].shape)  # delta_2  24x24
        self.grad_model["w1"] = nnu.convolution_gradient(delta["2"], self.z["h1"])
        
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


