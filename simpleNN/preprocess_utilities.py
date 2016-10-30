
import numpy as np
import nn_train as nnt


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



