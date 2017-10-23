
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


    # X is the of shape m X time_steps
    # prev_hidden m X hidden_size
    def forward(self, X, prev_hidden, time_step):
        update_gate = self.updateGate(X, self.Whz, self.Uiz, self.bhz, self.biz, prev_hidden, time_step, self.sigmoid)
        reset_gate = self.resetGate(X, self.Whr, self.Uir, self.bhr, self.bir, prev_hidden, time_step, self.sigmoid)
        memory_gate = self.memoryGate(X, self.Wg, self.Ug, self.bg, self.big, reset_gate, prev_hidden, time_step, self.tanh)
        current_hidden = update_gate * memory_gate + (1-update_gate) * prev_hidden
        out = self.b + np.dot(current_hidden,self.V.T)
        output = self.softmax(out)
        cache = dict(update_gate=update_gate, reset_gate = reset_gate, memory_gate = memory_gate, current_hidden = current_hidden, output=output)
        return cache

    @staticmethod
    def updateGate(X, W, U , bh, bi, prev_hidden, t, sig):
        from_hidden = np.dot(prev_hidden, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return sig(from_hidden + from_input)

    @staticmethod
    def resetGate(X, W, U, bh, bi, prev_hidden, t, sig):
        from_hidden = np.dot(prev_hidden, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return sig(from_hidden + from_input)

    @staticmethod
    def memoryGate(X, W, U, bh, bi, reset_gate, prev_hidden, t, tanh):

        inter = reset_gate * prev_hidden
        from_hidden = np.dot(inter, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return tanh(from_hidden + from_input)


    @staticmethod
    def sigmoid(z):
        #receives m X hidden_nodes
        l = len(z)
        z -= np.max(z,axis=-1).reshape(l,1)
        return 1.0/(1 + np.exp(-z))


    @staticmethod
    def tanh(z):
        return np.tanh(z)


    @staticmethod
    def softmax(outputs):
        l = len(outputs)
        outputs -= np.max(outputs,axis=-1).reshape(l,1) #for numeric stability
        expo = np.exp(X)
        return 1.0*expo/np.sum(expo,axis=-1).reshape(l,1)



if __name__ == '__main__':
    obj = preprocess()
    data = obj.load()
    X = np.array(list(data.X_train[:])).astype(int)[:10]
    y = np.array(list(data.y_train[:])).astype(int)[:10]
    cell = GRU()
    prev_hidden = np.zeros((X.shape[0],cell.hidden_nodes))
    cache = cell.forward(X, prev_hidden, 0)
    print cache
