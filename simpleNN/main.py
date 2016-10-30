# __author__ = 'linkaixiang'
import numpy as np
from configure import FLAGS
import nn_train as nnt
from nn_train import NesterovAcceleratedGrad
from nn_train import StochasticGradientDescent
from nn_train import AdamGrad
from nn_train import Adamdelta
from NeuralNet import NeuralNet
import nn_utilities as nnu
import scipy.io as sio
import preprocess_utilities as pu



def readdata():
    """
    Read data from file and return train valid test data
    :return:
    """
    mattrain = sio.loadmat(FLAGS.inputdata_dir + FLAGS.inputdata_train)

    train_x = mattrain["images"]
    train_y = mattrain["labels"]
    train_datasize = len(train_y)
    train_x = np.array([train_x[ii].flatten() for ii in range(train_datasize)])

    train_x, train_y, valid_x, valid_y = pu.split_train_valid(train_x, train_y, FLAGS.valid_rate)

    mattest = sio.loadmat(FLAGS.inputdata_dir + FLAGS.inputdata_test)

    test_x = mattest["images"]
    test_y = mattest["labels"]
    test_datasize = len(test_y)
    test_x = np.array([test_x[ii].flatten() for ii in range(test_datasize)])

    return tuple([train_x, train_y, valid_x, valid_y, test_x, test_y])



def run_main(datatuple):
    """ performan training evalutaion and testing

    :param datatuple:
    :return:
    """
    train_x, train_y, valid_x, valid_y, test_x, test_y = datatuple


    max_iteration = FLAGS.max_iteration
    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes
    nn_model = NeuralNet(n_layer, n_nodes, n_feat, FLAGS.func_num)
    nn_model = StochasticGradientDescent(datatuple, nn_model)

    print "test accuracy: ", nnt.evaluate_accuracy(test_x, test_y, nn_model)




def main():
    ''' Main function
    '''

    data_tuple = readdata()

    run_main(data_tuple)



if __name__ == '__main__':
    """ Main function, read data, train model
    """

    main()

# def test()
# # random test data
# features = np.random.randn(FLAGS.data_size, FLAGS.n_feat)
# labels = np.random.randn(FLAGS.data_size, FLAGS.n_nodes[-1])
# NesterovAcceleratedGrad(features, labels, FLAGS.Nesterov_alpha, FLAGS.Nesterov_beta)
# StochasticGradientDescent(features, labels)
# AdamGrad(features, labels)
# Adamdelta(features, labels, FLAGS.adadelta_gamma)
