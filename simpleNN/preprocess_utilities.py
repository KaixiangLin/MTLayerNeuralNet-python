
import numpy as np
import nn_train as nnt


def check_data_valid(dataset_tuple):
    print "checking data valid"
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataset_tuple
    if np.isnan(np.sum(train_x)) or np.isnan(np.sum(train_y)) or np.isnan(np.sum(valid_x)):
        print('\n data is no valid 1\n')
    if np.isnan(np.sum(valid_y)) or np.isnan(np.sum(test_x)) or np.isnan(np.sum(test_y)):
        print('\n data is no valid 2\n')

def normalize_data(x):
    """ normalize data to zero mean, variance 1.

    :param x:
    :return:
    """
    train_mean = np.mean(x, axis=0)
    train_std  = np.std(x, axis=0)

    train_std[train_std == 0] = 1
    x_new = (x - train_mean)/train_std

    return x_new



def split_train_valid(x, y, valid_rate):
    '''Split data set into training, validation'''

    data_size = len(y)
    num_valid = int(data_size * valid_rate)

    index = np.random.permutation(data_size)

    valid_index = index[:num_valid]
    train_index = index[num_valid:]

    train_x = x[train_index, :]
    train_y = y[train_index]

    valid_x = x[valid_index, :]
    valid_y = y[valid_index]

    return train_x, train_y, valid_x, valid_y

def Optimizers(flags):
    """ Return function handle as

    :param flags:
    :return:
    """
    if flags == "StochasticGradientDescent":
        return nnt.StochasticGradientDescent
    elif flags == "NesterovAcceleratedGrad":
        return nnt.NesterovAcceleratedGrad
    else:
        print "I didn't write this optimizer..."



