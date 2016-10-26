# __author__ = 'linkaixiang'
import scipy.io as sio
from NeuralNet import NeuralNet
import numpy as np

def train(batch_x, batch_y, nn_model):

    batch_size = len(batch_y)

    grad_temp = {k: np.zeros_like(v) for k, v in nn_model.model.iteritems()}

    for ii in range(batch_size):
        y_pred = nn_model.feedforward(batch_x[ii,:].T)
        # obj_val = nn_model.loss_function_l2(y_pred, batch_y[ii])
        y_true = batch_y[ii]
        err = y_pred - y_true
        grad_delta = nn_model.backprop(err).iteritems()
        for k, v in grad_delta:
            grad_temp[k] += v

        # gradient descend



def run_train(features, labels):

    max_iteration = 10
    n_layer = 3  # the last layer is the output of network
    n_feat = 10
    n_nodes = [3, 3, 1]

    nn_model = NeuralNet(n_layer,n_nodes, n_feat)
    for i in range(max_iteration):
        train(features, labels, nn_model)



def main():
    '''Main function'''

    # read data
    fname = "./problem-4.mat"
    mat_contents = sio.loadmat(fname)

    dataset_tuple = tuple([mat_contents['x'], mat_contents['y']])
    features, labels = dataset_tuple

    run_train(features, labels)



if __name__ == '__main__':
    main()

