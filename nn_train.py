# __author__ = 'linkaixiang'
import scipy.io as sio
from NeuralNet import NeuralNet
import nn_optimizer as opt
import numpy as np
from configure import FLAGS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_loss(batch_x, batch_y, nn_model):
    ''' evaluate the loss value over the batch
    :param batch_x:
    :param batch_y:
    :param nn_model:
    :return:
    '''
    batch_size = len(batch_y)
    temp_loss = 0
    for ii in range(batch_size):
        y_pred = nn_model.feedforward(batch_x[ii].T)
        temp_loss += nn_model.loss_function_l2(y_pred, batch_y[ii])

    return temp_loss


def train(batch_x, batch_y, nn_model):
    """ Apply one step gradient descent
    :param batch_x:
    :param batch_y:
    :param nn_model:
    :return:
    """

    batch_size = len(batch_y)

    delta_grad = {k: np.zeros_like(v) for k, v in nn_model.model.iteritems()}

    for ii in range(batch_size):
        y_pred = nn_model.feedforward(batch_x[ii].T)
        # obj_val = nn_model.loss_function_l2(y_pred, batch_y[ii])
        y_true = batch_y[ii]
        err = nn_model.loss_function_l2_grad(y_pred, y_true)
        grad_delta = nn_model.backprop(err).iteritems()
        for k, v in grad_delta:
            delta_grad[k] += v

    # gradient descend
    opt.GradientDescentOptimizer(nn_model, delta_grad, FLAGS.learning_rate)



def run_train(features, labels):

    max_iteration = FLAGS.max_iteration
    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes

    nn_model = NeuralNet(n_layer, n_nodes, n_feat)

    loss_val = []
    for i in range(max_iteration):
        train(features, labels, nn_model)
        loss_val.append(evaluate_loss(features, labels, nn_model))

    fig = plt.figure(1)
    plt.plot(loss_val)
    plt.savefig('./objvalue.png')
    plt.close(fig)


def main():
    ''' Main function
    '''

    # read data
    features = np.random.randn(FLAGS.batch_size, FLAGS.n_feat)
    labels = np.random.randn(FLAGS.batch_size, FLAGS.n_nodes[-1])

    run_train(features, labels)



if __name__ == '__main__':
    main()

