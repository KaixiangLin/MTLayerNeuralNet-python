



class myFLAGS:

    def __init__(self):
        ''' Initialization of network hyper parameters
        '''

        '''Network structure parameters'''
        self.n_feat = 784
        self.n_nodes = [32, 32, 32, 1]  # h2, h3, h4; where h1 is input x, h4 is output predicted value
        self.n_layer = len(self.n_nodes)

        '''Optimizer parameters'''
        self.Optimizer = "NesterovAcceleratedGrad"
        # StochasticGradientDescent; NesterovAcceleratedGrad;
        self.learning_rate = 1e-3
        self.Nesterov_alpha = 0.9
        self.Nesterov_beta = 0.1
        self.adadelta_gamma = 0.9

        '''data parameters'''
        self.inputdata_dir = "../data/"
        self.inputdata_test = "mnistTest.mat"
        self.inputdata_train = "mnistTrain.mat"
        self.data_size = 50000

        '''Training parameters'''
        self.valid_rate = 1.0/6
        self.num_epoch = 1
        self.func_num = 1  # 1 relu, 2 sigmoid, 3 tanh
        self.batch_size = 100
        self.max_iteration = 500 # int(self.data_size * self.num_epoch / self.batch_size)


FLAGS = myFLAGS()