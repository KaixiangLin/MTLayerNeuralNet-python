import numpy as np
import scipy.io as sio
import nn_train
import NeuralNet as nn
from configure import FLAGS

def dict_to_nparray(grad_dictionary):
    """ Convert dictionary to a numpy array checked

    :param grad_dictionary:
    :param n_nodes:
    :return:
    """

    x = []
    for ii in range(FLAGS.n_layer):
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


def gradientCheck(gradfunc, objfunc, x0, X, y,  num_checks, nn_model):
    """ Check the gradient

    :param gradfunc: gradient function handle
    :param objfunc: objective function handle
    :param x0: initialization point, an one dimension vector numpy array, representing parameter
    :param num_checks: how many times you check it
    :return:
    """
    epsilon = 1e-7
    sum_error = 0
    for ii in range(num_checks):
        delta = 0

        J = np.random.choice(range(len(x0)))   # choose a specific variable to check its gradient
        delta = np.random.randn(1)[0] * epsilon
        x1 = np.copy(x0)
        x2 = np.copy(x0)
        x1[J] = x0[J] + delta
        x2[J] = x0[J] - delta
        g = gradfunc(x0, X, y, nn_model)

        f1 = objfunc(x1, X, y, nn_model)
        f2 = objfunc(x2, X, y, nn_model)

        g_est = (f1 - f2)/(2 * delta)

        error = abs(g[J] / g_est)

        print '% 5d  % 6d % 15g % 15f % 15f\n' %(ii, J, error, g[J], g_est)
        sum_error += error
    return sum_error

def objfunc(x1, X, y, nn_model):
    """ Calculate the objective function value

    :param x1: the current weights in Neural network
    :param X:  the input data
    :param y: the label
    :param nn_model: instance of neural network class
    :return:
    """
    model_w = nparray_to_dictionary(x1, FLAGS.n_feat, FLAGS.n_nodes, FLAGS.n_layer)
    nn_model.model = model_w

    f = nn_train.evaluate_loss(X, y, nn_model)

    return f

def gradfunc(x0, X, y, nn_model):
    """ Calculate gradient for neural network, converting input to diction format

    :param x0:
    :param X: one input instance
    :param y: label corresponding to the input
    :return:
    """
    w = nparray_to_dictionary(x0, FLAGS.n_feat, FLAGS.n_nodes, FLAGS.n_layer)
    nn_model.model = w
    w_new = np.zeros_like(x0)
    batch_size = len(y)
    for ii in range(batch_size):
        y_pred = nn_model.feedforward(X[ii].T)
        y_true = y[ii]
        err = nn_model.loss_function_l2_grad(y_pred, y_true)
        grad_delta = nn_model.backprop(err)
        w_temp = dict_to_nparray(grad_delta)
        w_new += w_temp

    return w_new


def run_main(features, labels):

    max_iteration = FLAGS.max_iteration
    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes

    nn_model = nn.NeuralNet(n_layer, n_nodes, n_feat, FLAGS.func_num)
    nn_train.train(features, labels, nn_model)

    w0 = dict_to_nparray(nn_model.model)

    print gradientCheck(gradfunc, objfunc, w0, [features[1]], [labels[1]], max_iteration, nn_model)

def main():

    # read data
    # fname = "../problem-4.mat"
    # mat_contents = sio.loadmat(fname)
    #
    # dataset_tuple = tuple([mat_contents['x'], mat_contents['y']])
    # features, labels = dataset_tuple
    features = np.random.randn(100, FLAGS.n_feat)
    labels = np.random.randn(100, FLAGS.n_nodes[-1])

    run_main(features, labels)

if __name__ == "__main__":
    main()




