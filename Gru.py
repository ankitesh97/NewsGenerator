
import numpy as np
import pickle
from preprocessor import preprocess
import json
from joblib import Parallel, delayed
import time
import multiprocessing
import sys

p_file = open('params.json','r')
params = json.loads(p_file.read())["gru"]


class GRU:

    #update gate(z), reset gate(r), memory gate(g)
    def __init__(self):
        self.Uiz = None # input to update gate
        self.biz = None
        self.Whz = None # hidden to update gate
        self.bhz = None
        self.Uir = None # input to reset gate
        self.bir = None
        self.Whr = None # hidden to reset gate
        self.bhr = None
        self.Wg = None # for gate gate
        self.bg = None
        self.Ug = None # input to gate gate
        self.big = None
        self.gate_params_shape = (params["hidden_nodes"],params["hidden_nodes"])
        self.word_dim = params['preprocess']['vocab_size']
        self.randomizeParams()


    # X is the of shape m X time_steps
    # prev_hidden m X hidden_size
    def forward(self, X, prev_hidden, time_step):
        update_gate = self.updateGate(X, self.Whz, self.Uiz, self.bhz, self.biz, prev_hidden, time_step)
        
        pass

    @staticmethod
    def updateGate(X, W, U , bh, bi, prev_hidden, t):
        from_hidden = np.dot(prev_hidden, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return self.sigmoid(from_hidden + from_input)



    @staticmethod
    def sigmoid(z):
        #receives m X hidden_nodes
        l = len(z)
        z -= np.max(z,axis=-1).reshape(l,1)
        return 1.0/(1 + np.exp(-z))
