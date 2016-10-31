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

        self.n_feat = 784
        self.n_nodes = [32, 32, 32, 32, 32, 1]  # h2, h3, h4; where h1 is input x, h4 is output predicted value
        self.n_layer = len(self.n_nodes)

        '''Optimizer parameters'''
        self.Optimizer = 3 #1: SGD 2: "NesterovAcceleratedGrad" 3: Adam 4 Adamdelta
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
        self.func_num = 1  # 1 relu, 2 sigmoid, 3 tanh
        self.batch_size = 128
        self.max_iteration = int(5e+4 + 1) # int(self.data_size * self.num_epoch / self.batch_size)
        self.record_persteps = 1000  # every x steps records the models

    def create_dir(self):

        # if not os.path.exists(FLAGS.fig_dir):
        #     os.makedirs(FLAGS.fig_dir)
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)


FLAGS = myFLAGS()