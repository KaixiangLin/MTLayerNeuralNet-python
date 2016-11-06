import os
import time
current_time = time.strftime("%Y%m%d_%H-%M")


class myFLAGS:

    def __init__(self):
        ''' Initialization of network hyper parameters
        '''

        self.current_time = current_time
        print current_time

        '''Network structure parameters'''
        self.n_feat = 28    # add bias term
        filter1_channel, filter2_channel, n_node_linear1_row = 32, 64, 200
        # 32, 64, 200 # for test 2, 2, 20

        self.maxpool_size = 2
        self.filter1_size = 5
        self.filter1_channel = filter1_channel  # todo should be 32    2
        self.filter2_size = 5
        self.filter2_channel = filter2_channel # todo should be 64   2

        self.n_node_linear1_row = n_node_linear1_row # todo should be 200   20
        # self.n_node_linear1_column = 1024  #4*4*64 todo testing should be 1024 = 4*4*64    32

        '''Optimizer parameters'''
        self.Optimizer = 4 #1: SGD 2: "NesterovAcceleratedGrad" 3: AdamGrad 4 Adamdelta
        self.learning_rate = 1e-3
        self.Nesterov_alpha = 0.9
        self.adadelta_gamma = 0.9

        '''data parameters'''
        self.inputdata_dir = "../../data/"
        self.inputdata_test = "mnistTest.mat"
        self.inputdata_train = "mnistTrain.mat"
        self.fig_dir = "../results/" + current_time + "/"
        self.model_dir = "../results/" + current_time + "/"
        self.data_size = 50000

        '''Training parameters'''
        self.valid_rate = 1.0/6
        self.num_epoch = 1
        self.func_num = 3  # 1 relu, 2 sigmoid, 3 tanh
        self.batch_size = 128
        self.max_iteration = int(5e+3 + 1) # int(self.data_size * self.num_epoch / self.batch_size)
        self.record_persteps = 1000 # every x steps records the models
        self.batch_size_evaluate = 1000
        self.nnmodel_load_fname = "../results/20161106_09-39/Adamdelta_189.npy"
        self.mnist_input = "../../data/mnist_split_images.npy"  # each input is a 28*28 image

    def create_dir(self):

        # if not os.path.exists(FLAGS.fig_dir):
        #     os.makedirs(FLAGS.fig_dir)
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)


FLAGS = myFLAGS()