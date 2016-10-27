



class myFLAGS:

    def __init__(self):
        ''' Initialization of network hyper parameters
        '''

        self.n_feat = 10
        self.n_nodes = [3, 3, 3, 10]  # h2, h3, h4; where h1 is input x, h4 is output predicted value
        self.n_layer = len(self.n_nodes)

        self.learning_rate = 1e-3
        self.max_iteration = 100

FLAGS = myFLAGS()