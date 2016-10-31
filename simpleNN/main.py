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
from shutil import copyfile

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
    train_x = pu.normalize_data(train_x)

    train_x, train_y, valid_x, valid_y = pu.split_train_valid(train_x, train_y, FLAGS.valid_rate)

    mattest = sio.loadmat(FLAGS.inputdata_dir + FLAGS.inputdata_test)

    test_x = mattest["images"]
    test_y = mattest["labels"]
    test_datasize = len(test_y)
    test_x = np.array([test_x[ii].flatten() for ii in range(test_datasize)])
    test_x = pu.normalize_data(test_x)

    return tuple([train_x, train_y, valid_x, valid_y, test_x, test_y])


def hidden_layer_crossvalidation(datatuple):
    """ Cross validation to choose the number of hidden layers"""
    FLAGS.create_dir()
    # save configuration
    configure_name = "configure.py"
    copyfile(configure_name, FLAGS.model_dir + configure_name)

    trainaccs = []
    validaccs = []

    for i in [3, 4, 5, 6]:
        FLAGS.n_nodes = [32] * i + [1]
        FLAGS.n_layer = len(FLAGS.n_nodes)
        print FLAGS.n_nodes
        nn_model = NeuralNet(FLAGS.n_layer, FLAGS.n_nodes, FLAGS.n_feat, FLAGS.func_num)
        nn_model = Adamdelta(datatuple, nn_model, FLAGS.adadelta_gamma)
        trainaccs.append(nn_model.train_acc)
        validaccs.append(nn_model.valid_acc)
        nnu.save_model(nn_model, FLAGS.model_dir + "Adamdelta_" + str(FLAGS.max_iteration) + ".p")

    print "final round training accuracy: ", trainaccs
    print "final round validation accuracy: ", validaccs
    results = tuple([trainaccs, validaccs])
    np.save(FLAGS.model_dir + "results.npy", results)

def run_main(datatuple):
    """ performan training evalutaion and testing

    :param datatuple:
    :return:
    """
    train_x, train_y, valid_x, valid_y, test_x, test_y = datatuple
    FLAGS.create_dir()

    # create network model
    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes
    nn_model = NeuralNet(n_layer, n_nodes, n_feat, FLAGS.func_num)

    # save configuration
    configure_name = "configure.py"
    copyfile(configure_name, FLAGS.model_dir + configure_name)
    if FLAGS.Optimizer == 1:
        print "Optimizer: SGD \n"
        nn_model = StochasticGradientDescent(datatuple, nn_model)
        nnu.save_model(nn_model, FLAGS.model_dir + "SGD_" + str(FLAGS.max_iteration))

    elif FLAGS.Optimizer == 2:
        print "Optimizer: NesterovAcceleratedGrad \n"
        nn_model = NesterovAcceleratedGrad(datatuple, nn_model, FLAGS.Nesterov_alpha, FLAGS.learning_rate)
        nnu.save_model(nn_model, FLAGS.model_dir + "Nesterov_" + str(FLAGS.max_iteration))
    elif FLAGS.Optimizer == 3:
        print "Optimizer: AdamGrad \n"
        nn_model = AdamGrad(datatuple, nn_model)
        nnu.save_model(nn_model, FLAGS.model_dir + "AdamGrad_" + str(FLAGS.max_iteration))
    elif FLAGS.Optimizer == 4:
        print "Optimizer: Adamdelta \n"
        nn_model = Adamdelta(datatuple, nn_model, FLAGS.adadelta_gamma)
        nnu.save_model(nn_model, FLAGS.model_dir + "Adamdelta_" + str(FLAGS.max_iteration))

    print "test accuracy: ", nnt.evaluate_accuracy(test_x, test_y, nn_model)


def test():
    # random test data
    # testing
    FLAGS.n_feat = 3
    FLAGS.n_nodes = [32, 32, 32, 32, 1]
    FLAGS.data_size_test = 500
    FLAGS.n_layer = len(FLAGS.n_nodes)

    features = np.random.randn(FLAGS.data_size_test, FLAGS.n_feat)
    labels = np.random.randn(FLAGS.data_size_test, FLAGS.n_nodes[-1])

    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes
    nn_model = NeuralNet(n_layer, n_nodes, n_feat, FLAGS.func_num)
    # nnu.save_model(nn_model, FLAGS.model_dir + "test.p")
    # nn_model2 = nnu.load_model(FLAGS.model_dir + "test.p")
    # nn_temp_dict = nnu.dict_minus(nn_model.model, nn_model2.model)

    datatuple = tuple([features, labels, features, labels, features, labels])
    NesterovAcceleratedGrad(datatuple, nn_model, FLAGS.Nesterov_alpha, FLAGS.learning_rate)
    # StochasticGradientDescent(datatuple, nn_model)
    # AdamGrad(datatuple, nn_model)
    # Adamdelta(datatuple, nn_model, FLAGS.adadelta_gamma)



def main():
    ''' Main function
    '''

    # test()

    data_tuple = readdata()

    run_main(data_tuple)


    # hidden_layer_crossvalidation(data_tuple)


if __name__ == '__main__':
    """ Main function, read data, train model
    """

    main()


