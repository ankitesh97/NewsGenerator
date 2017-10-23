
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

class GRUCell:

    #update gate(z), reset gate(r), memory gate(g)
    def __init__(self):
        self.cache = None
        self.errors = None



    # X is the of shape m X time_steps
    # prev_hidden m X hidden_size
    def forward(self, X, prev_hidden, time_step, weights):
        update_gate = self.updateGate(X, weights["Whz"], weights["Uiz"], weights["bhz"], weights["biz"], prev_hidden, time_step, self.sigmoid)
        reset_gate = self.resetGate(X, weights["Whr"], weights["Uir"], weights["bhr"], weights["bir"], prev_hidden, time_step, self.sigmoid)
        memory_gate = self.memoryGate(X, weights["Wg"], weights["Ug"], weights["bg"], weights["big"], reset_gate, prev_hidden, time_step, self.tanh)
        current_hidden = update_gate * memory_gate + (1-update_gate) * prev_hidden
        out = self.b + np.dot(current_hidden,weights["V"].T)
        output = self.softmax(out)
        self.cache = dict(update_gate=update_gate, reset_gate = reset_gate, memory_gate = memory_gate, current_hidden = current_hidden, output=output)

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



    def backprop(self, cache):
        pass

if __name__ == '__main__':
    obj = preprocess()
    data = obj.load()
    X = np.array(list(data.X_train[:])).astype(int)[:10]
    y = np.array(list(data.y_train[:])).astype(int)[:10]
    cell = GRUCell()
    prev_hidden = np.zeros((X.shape[0],cell.hidden_nodes))
    cache = cell.forward(X, prev_hidden, 0)
    print cache
