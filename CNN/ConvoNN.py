# __author__ = 'linkaixiang'

import numpy as np
import nn_utilities as nnu
import math
class NeuralNet:

    def __init__(self, n_feat, func_num, filter1_channel, filter2_channel,
                 n_node_linear1_row):
        ''' @:param: n_layer: number of layers
            @:param: n_nodes: list of numbers e.g. [1,2,3], if it's 3 layers and 1,2,3 hidden nodes respectively
            @:param: n_feat: the number of input features
            @:param: w start from 1
            @:param: h start from 1, h_1 is input value
        '''

        self.train_acc = 0
        self.valid_acc = 0
        # self.n_layer = 2
        self.n_feat = n_feat
        self.func_num = func_num
        self.model = {}
        self.hidden = {}
        self.z = {}  # input to the activation function
        self.maxpool = {}
        self.grad_model = {}

        self.maxpool_size = 2
        self.filter1_size = 5
        self.filter1_channel = filter1_channel  # todo should be 32    2
        self.filter2_size = 5
        self.filter2_channel = filter2_channel # todo should be 64   2

        self.n_node_linear1_row = n_node_linear1_row # todo should be 200   20
        temp_a = ((n_feat - self.filter1_size + 1)/self.maxpool_size - self.filter2_size + 1)/self.maxpool_size
        self.n_node_linear1_column = temp_a**2*self.filter2_channel  #4*4*64 todo testing should be 1024 = 4*4*64    32

        self.n_node_linear2_row = 10 # should be same as class to distinguish
        self.n_node_linear2_column = self.n_node_linear1_row

        print "n_node_linear1_column", self.n_node_linear1_column
        # convolution filter 5x5 32
        self.model["w1"] = np.random.randn(self.filter1_size, self.filter1_size, self.filter1_channel) \
                           / np.sqrt(math.sqrt(float(self.filter1_size)))
        print "w1 init: ", self.model["w1"].shape
        # convolution filter 5x5 64
        self.model["w2"] = np.random.randn(self.filter2_size, self.filter2_size, self.filter2_channel) \
                           / np.sqrt(math.sqrt(float(self.filter2_size)))
        print "w2 init: ", self.model["w2"].shape

        # linear
        self.model["w3"] = np.random.randn(self.n_node_linear1_row, self.n_node_linear1_column) \
                            / np.sqrt(math.sqrt(float(self.n_node_linear1_row)))
        print "w3 init: ", self.model["w3"].shape
        self.conv2linear = np.zeros((self.n_node_linear1_column, 1))

        # last layer
        self.model["w4"] = np.random.randn(self.n_node_linear2_row, self.n_node_linear2_column) \
                            / np.sqrt(math.sqrt(float(self.n_node_linear2_row)))
        print "w4 init: ", self.model["w4"].shape
        # self.hidden["h2"] = np.zeros((self.n_feat - self.filter1_size + 1, self.n_feat - self.filter1_size + 1))

        self.grad_model = {k: np.zeros_like(v) for k, v in self.model.iteritems()}

    def activation(self, y, func_num):
        ''' @:param w:   m x n numpy array
            @:param x:   n x 1 numpy array
        '''

        # y = np.dot(w, x)
        h = nnu.activation_funcs(func_num, y)
        return h


    def feedforward(self, x):
        ''' Define CNN network

        :param x: input data 28 * 28 image
        :return:
        '''

        self.hidden["h1"] = x  # output of activation function
        self.z["h1"] = x  # input of activation function

        # convolution layer 5 by 5, 32
        self.z["h2"] = nnu.convolution_tensor(self.hidden["h1"], self.model["w1"])
        #              output tensor after convo:  24x24xC

        # tanh
        self.hidden["h2"] = self.activation(self.z["h2"], self.func_num)

        # maximum pooling
        self.maxpool["h2"], self.maxpool["h2_pos"] = nnu.max_pooling(self.hidden["h2"], self.maxpool_size) #12x12,32

        # convolution layer (5, 5, 64)
        self.z["h3"] = nnu.convolution_tensor_tensor(self.maxpool["h2"], self.model["w2"])

        # tanh
        self.hidden["h3"] = self.activation(self.z["h3"], self.func_num)

        # max pooling
        self.maxpool["h3"], self.maxpool["h3_pos"] = nnu.max_pooling(self.hidden["h3"], self.maxpool_size)
        # self.maxpool["h3_pos"] has same size as self.hidden["h3"], is a 0/1 mask
        # self.maxpool["h3"]  4x4x64

        # reshape output into a vector, the input of fully connect layer, 1024x1
        self.conv2linear = self.maxpool["h3"].reshape(self.maxpool["h3"].size, 1)

        # z4 = W*h3
        self.z["h4"] = np.dot(self.model["w3"], self.conv2linear)

        # h4 = tanh(z4)   200x1
        self.hidden["h4"] = self.activation(self.z["h4"], self.func_num)

        # z5 = w4 h4
        self.z["h5"] = np.dot(self.model["w4"], self.hidden["h4"])

        # h5 = softmax
        self.hidden["h5"] = nnu.activate_softmax(self.z["h5"])

        # self.con2linea = self.hidden["h2"]  # last layer before softmax, i.e. the h in softmax(wh + b)
        # z1, z2, z3 = self.z["h2"].shape
        #soft max layer
        # self.z["h3"] = np.dot(self.model["w2"], self.hidden["h2"])
        # self.hidden["h3"] = nnu.activate_softmax(self.z["h"+str(self.n_layer+1)])

        return self.hidden["h5"]  # return last hidden layer


    def backprop(self, err):
        ''' back propagation
        :return: the gradient with respect to the error.
        '''

        delta = {}
        # last layer gradient delta (error)

        delta[str(5)] = err # delta_3  10 x 1
        self.grad_model["w4"] = np.outer(delta["5"], self.hidden["h4"].T) # 10 x 200

        delta_w4 = np.dot(self.model["w4"].T, delta["5"])  #  (200)x10 dot 10x1 = 200x1
        delta["4"] = np.multiply(delta_w4, nnu.grad_activation(self.func_num, self.z["h4"]))   # delta_4 200 x 1

        self.grad_model["w3"] = np.outer(delta["4"], self.conv2linear.T) # 200 x 1024

        # ERROR prop to w2
        delta_w3 = np.dot(self.model["w3"].T, delta["4"])   #1024x1
        delta["3"] = delta_w3.reshape(self.maxpool["h3"].shape) # delta of con2linear 4x4x64

        delta_maxpool3 = nnu.max_pool_inv(self.maxpool["h3_pos"],  delta["3"], self.maxpool_size)  # err, 8x8x64
        delta_tanh = delta_maxpool3 * nnu.grad_activation(self.func_num, self.z["h3"]) # err, 8x8x64

        self.grad_model["w2"] = nnu.convolution_gradient(delta_tanh, np.sum(self.maxpool["h2"], axis=2))

        # error prop to w1
        delta_w2 = nnu.convolution_backprop(delta_tanh, self.model["w2"])
        delta["2"] = np.repeat(delta_w2[:, :, np.newaxis], self.filter1_channel, axis=2)  # 12x12,32

        delta_maxpool2 = nnu.max_pool_inv(self.maxpool["h2_pos"], delta["2"], self.maxpool_size)  # 24x24, 32
        delta_tanh2 = delta_maxpool2 * nnu.grad_activation(self.func_num, self.z["h2"])  # 24x24, 32

        self.grad_model["w1"] = nnu.convolution_gradient(delta_tanh2, self.z["h1"])

        # m1, _, C1 = self.z["h2"].shape
        # delta["2"] = nnu.reshape_vec_tensor(delta_w3, m1, C1)  # delta_2  24x24
        # self.grad_model["w1"] = nnu.convolution_gradient(delta["2"], self.z["h1"])

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


