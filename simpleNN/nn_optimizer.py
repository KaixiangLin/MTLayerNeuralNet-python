#



def GradientDescentOptimizer(nn_model, delta_grad, learning_rate):
    '''

    :param nn_model: neural network instance
    :param delta_grad: changing of gradient from current batch
    :param learning_rate: step size
    :return:
    '''

    for k, v in nn_model.model.iteritems():

        nn_model.model[k] -= learning_rate * delta_grad[k]



