#__author__ = 'linkaixiang'

import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh((0, np.pi*1j, np.pi*1j/2))

def activation_funcs(func_num, x):
    """ @:param func_num: choose different type of activate functions
        func_num = 1: relu
    """
    output = 0

    if func_num == 1:
        output = x[x < 0] = 0
    elif func_num == 2:
        output = sigmoid(x)
    elif func_num == 3:
        output = np.tanh(x)
    else:
        print("ERROR WRONG Activation function")
        pass

    return output

def grad_activation(func_num, x):

    gradient = 0
    if func_num == 1:
        gradient = x
        gradient[gradient<=0] = 0
        gradient[gradient>0] = 1

    elif func_num == 2:
        gradient = np.multiply(sigmoid(x), 1 - sigmoid(x))

    elif func_num == 3:
        gradient = 1 - tanh(x) ** 2
    else:
        print("ERROR WRONG GRADIENT")

    return gradient
