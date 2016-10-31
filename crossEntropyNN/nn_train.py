# __author__ = 'linkaixiang'
import scipy.io as sio
from NeuralNet import NeuralNet
import nn_optimizer as opt
import numpy as np
from configure import FLAGS
import nn_utilities as nnu



def evaluate_accuracy(batch_x, batch_y, nn_model):
    ''' evaluate the prediction accuracy
    :param batch_x:
    :param batch_y:
    :param nn_model:
    :return:
    '''
    batch_size = len(batch_y)
    temp_loss = 0
    count = 0
    for ii in range(batch_size):
        y_pred = nn_model.feedforward(batch_x[ii].T)
        if int(round(y_pred)) == batch_y[ii]:
            count += 1


    return count / float(batch_size)


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


def compute_gradient(batch_x, batch_y, nn_model):
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
        y_true = batch_y[ii]
        err = nn_model.loss_function_l2_grad(y_pred, y_true)
        grad_delta = nn_model.backprop(err).iteritems()
        for k, v in grad_delta:
            delta_grad[k] += v / float(batch_size)

    return delta_grad



def StochasticGradientDescent(datatuple, nn_model):
    """  gradient descent over the batch features

    :param datatuple:
    :return:
    """
    train_x, train_y, valid_x, valid_y, _, _ = datatuple
    loss_val = []
    train_accs = []
    valid_accs = []
    for i in range(FLAGS.max_iteration):
        batch_x, batch_y = nnu.batch_data(train_x, train_y, FLAGS.batch_size)

        delta_grad = compute_gradient(batch_x, batch_y, nn_model)
        opt.GradientDescentOptimizer(nn_model, delta_grad, FLAGS.learning_rate )

        if i % FLAGS.record_persteps == 0:
            train_acc = evaluate_accuracy(train_x, train_y, nn_model)
            train_accs.append(train_acc)
            valid_acc = evaluate_accuracy(valid_x, valid_y, nn_model)
            valid_accs.append(valid_acc)
            print "step ", i, " training acc: ", train_acc, " valid acc:", valid_acc
            loss_val.append(evaluate_loss(train_x, train_y, nn_model))

    nnu.plot_list(loss_val, FLAGS.fig_dir + "SGD_objvalue.png", 1)
    nnu.plot_list_acc(train_accs, valid_accs, FLAGS.fig_dir + "SGD_accs.png")
    nn_model.train_acc = train_accs[-1]
    nn_model.valid_acc = valid_accs[-1]
    return nn_model

def NesterovAcceleratedGrad(datatuple, nn_model, alpha, beta):
    """ Stochastic Gradient Descent with Nesterov Acceleration

    :param datatuple:
    :return:
    """
    train_x, train_y, valid_x, valid_y, _, _ = datatuple

    max_iteration = FLAGS.max_iteration
    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes
    #
    # nn_model = NeuralNet(n_layer, n_nodes, n_feat)
    train_accs = []
    valid_accs = []
    loss_val = []
    cumulative_grad_np = np.zeros_like(nnu.dict_to_nparray(nn_model.model, n_layer))
    cumulative_grad = nnu.nparray_to_dictionary(cumulative_grad_np, n_feat, n_nodes, n_layer)
    for i in range(max_iteration):

        batch_x, batch_y = nnu.batch_data(train_x, train_y, FLAGS.batch_size)

        cumulative_grad = nnu.dict_mulscala(cumulative_grad, alpha)     # alpha * vt-1
        nn_model.model = nnu.dict_add(nn_model.model, cumulative_grad)   # theta = theta0 + alpha * vt-1

        # compute gradient
        delta_grad = compute_gradient(batch_x, batch_y, nn_model)  # gradient of L(theta0 + alpha * vt-1)

        delta_grad = nnu.dict_mulscala(delta_grad, -beta)  # - beta * g
        cumulative_grad = nnu.dict_add(cumulative_grad, delta_grad)  # vt = alpha vt-1 - beta * g

        nn_model.model = nnu.dict_add(nn_model.model, delta_grad) #theta = theta0 + alpha * vt-1 - beta*g
        if i % FLAGS.record_persteps == 0:
            train_acc = evaluate_accuracy(train_x, train_y, nn_model)
            train_accs.append(train_acc)
            valid_acc = evaluate_accuracy(valid_x, valid_y, nn_model)
            valid_accs.append(valid_acc)
            print "step ", i, " training acc: ", train_acc, " valid acc:", valid_acc
            loss_val.append(evaluate_loss(train_x, train_y, nn_model))

    nnu.plot_list(loss_val, FLAGS.fig_dir + "Nesterov_objvalue.png", 1)
    nnu.plot_list_acc(train_accs, valid_accs, FLAGS.fig_dir + "Nesterov_accs.png")
    nn_model.train_acc = train_accs[-1]
    nn_model.valid_acc = valid_accs[-1]
    return nn_model

def AdamGrad(datatuple, nn_model):
    """ Adam Gradient optimizer

    :param datatuple:
    :return:
    """
    train_x, train_y, valid_x, valid_y, _, _ = datatuple
    max_iteration = FLAGS.max_iteration
    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes

    epsilon = 1e-8

    train_accs = []
    valid_accs = []
    loss_val = []
    G_np = np.zeros_like(nnu.dict_to_nparray(nn_model.model, n_layer))
    G = nnu.nparray_to_dictionary(G_np, n_feat, n_nodes, n_layer)
    for i in range(max_iteration):

        batch_x, batch_y = nnu.batch_data(train_x, train_y, FLAGS.batch_size)

        # compute gradient
        delta_grad = compute_gradient(batch_x, batch_y, nn_model)  # gradient of L(theta)

        g_square = nnu.dict_mul(delta_grad, delta_grad)   #  g^2

        G = nnu.dict_add(G, g_square)  # vt = alpha vt-1 - beta * g

        G_np = nnu.dict_to_nparray(G, n_layer)

        temp_np = np.divide(-float(FLAGS.learning_rate), np.sqrt(G_np + epsilon))   # -learning_rate * / sqrt(G_t + epsilon)

        temp_dict = nnu.nparray_to_dictionary(temp_np, n_feat, n_nodes, n_layer) # converet np array to dictionary

        temp_dict = nnu.dict_mul(temp_dict, delta_grad)

        nn_model.model = nnu.dict_add(nn_model.model, temp_dict)


        if i % FLAGS.record_persteps == 0:
            train_acc = evaluate_accuracy(train_x, train_y, nn_model)
            train_accs.append(train_acc)
            valid_acc = evaluate_accuracy(valid_x, valid_y, nn_model)
            valid_accs.append(valid_acc)
            print "step ", i, " training acc: ", train_acc, " valid acc:", valid_acc
            loss_val.append(evaluate_loss(train_x, train_y, nn_model))

    nnu.plot_list_acc(train_accs, valid_accs, FLAGS.fig_dir + "Adam_accs.png")
    nnu.plot_list(loss_val, FLAGS.fig_dir + "Adam_objvalue.png", 1)
    np.save(FLAGS.fig_dir + "Adam_accs.npy", tuple([train_accs, valid_accs]))
    nn_model.train_acc = train_accs[-1]
    nn_model.valid_acc = valid_accs[-1]
    return nn_model


def Adamdelta(datatuple, nn_model, gamma):
    """ Adamdelta Gradient optimizer

    :param datatuple:
    :return:
    """
    train_x, train_y, valid_x, valid_y, _, _ = datatuple
    max_iteration = FLAGS.max_iteration
    n_layer = FLAGS.n_layer  # the last layer is the output of network
    n_feat = FLAGS.n_feat
    n_nodes = FLAGS.n_nodes

    epsilon = 1e-8

    loss_val = []
    train_accs = []
    valid_accs = []

    G_np = np.zeros_like(nnu.dict_to_nparray(nn_model.model, n_layer))
    RMS_deltatheta_prev = np.zeros_like(G_np)
    Delta_theta = np.zeros_like(G_np)
    for i in range(max_iteration):

        batch_x, batch_y = nnu.batch_data(train_x, train_y, FLAGS.batch_size)

        # compute gradient
        delta_grad = compute_gradient(batch_x, batch_y, nn_model)  # gradient of L(theta)

        delta_grad_np = nnu.dict_to_nparray(delta_grad, n_layer)
        g_square_np = np.multiply(delta_grad_np, delta_grad_np) #  g^2

        G_np = G_np * gamma + g_square_np * (1 - gamma)  # Gt = gamma * Gt  + (1 - gamma) * g^2
        RMS_gt = np.sqrt(G_np + epsilon)  # sqrt(G_t + epsilon)

        delta_theta = np.multiply(-np.divide(RMS_deltatheta_prev, RMS_gt), delta_grad_np)  # - RMS_delta_theta^2 t-1 / RMS_g^2 t .* gt

        Delta_theta = Delta_theta*gamma + (delta_theta ** 2) * (1 - gamma)  # delta_theta^2*gamma + (1-gamma)*delta_theta^2

        RMS_theta = np.sqrt(Delta_theta + epsilon)
        RMS_deltatheta_prev = RMS_theta

        temp_dict = nnu.nparray_to_dictionary(delta_theta, n_feat, n_nodes, n_layer) # converet np array to dictionary
        nn_model.model = nnu.dict_add(nn_model.model, temp_dict)

        if i % FLAGS.record_persteps == 0:
            train_acc = evaluate_accuracy(train_x, train_y, nn_model)
            train_accs.append(train_acc)
            valid_acc = evaluate_accuracy(valid_x, valid_y, nn_model)
            valid_accs.append(valid_acc)
            print "step ", i, " training acc: ", train_acc, " valid acc:", valid_acc
            loss_val.append(evaluate_loss(train_x, train_y, nn_model))

    nnu.plot_list_acc(train_accs, valid_accs, FLAGS.fig_dir + "Adamdelta_accs.png")
    nnu.plot_list(loss_val, FLAGS.fig_dir + "Adamdelta_objvalue.png", 1)
    nn_model.train_acc = train_accs[-1]
    nn_model.valid_acc = valid_accs[-1]
    return nn_model

