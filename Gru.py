

import numpy as np
import pickle
from preprocessor import preprocess
import json
from joblib import Parallel, delayed
import time
import multiprocessing
import sys

p_file = open('params.json','r')
p = json.loads(p_file.read())
params = p["gru"]


class Gru:

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
        self.V = None #for the output
        self.b = None #for output
        self.hidden_nodes = params["hidden_nodes"]
        self.gate_params_shape = (params["hidden_nodes"],params["hidden_nodes"])
        self.word_dim = p['preprocess']['vocab_size']
        self.randomizeParams()




    def randomizeParams(self):
        self.Whz = np.random.randn(self.gate_params_shape[0],self.gate_params_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))
        self.Whr = np.random.randn(self.gate_params_shape[0],self.gate_params_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))
        self.Wg = np.random.randn(self.gate_params_shape[0],self.gate_params_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))

        self.Uiz = np.random.randn(self.gate_params_shape[1],self.word_dim) * np.sqrt(1.0/(1+self.word_dim))
        self.Uir = np.random.randn(self.gate_params_shape[1],self.word_dim) * np.sqrt(1.0/(1+self.word_dim))
        self.Ug = np.random.randn(self.gate_params_shape[1],self.word_dim) * np.sqrt(1.0/(1+self.word_dim))

        self.bhz = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes))
        self.bhr = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes))
        self.bg = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes))

        self.biz = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim))
        self.bir = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim))
        self.big = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim))

        self.V = np.random.randn(self.word_dim,self.gate_params_shape[1]) * np.sqrt(1.0/self.hidden_nodes+1)

        self.b = np.random.randn(self.word_dim) * np.sqrt(1.0/(1+self.word_dim))

        
