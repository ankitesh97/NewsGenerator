
import numpy as np
import pickle
from preprocessor import preprocess
import json
from joblib import Parallel, delayed
import time
import multiprocessing
import sys
# from Gru import Gru

p_file = open('params.json','r')
p = json.loads(p_file.read())
params = p["lstm"]

class LstmCell:

    def __init__(self,m):
        self.cache = None
        self.errors = None  # this will be initialized
        self.initErrors(m)



    def initErrors(self,m):
        hidden_nodes = params["hidden_nodes"]

        input_gate = np.zeros((m,hidden_nodes))
        forget_gate = np.zeros((m,hidden_nodes))
        gate_gate = np.zeros((m,hidden_nodes))
        output_gate = np.zeros((m,hidden_nodes))
        cell = np.zeros((m,hidden_nodes))
        hidden = np.zeros((m,hidden_nodes))
        output = np.zeros((m,p["preprocess"]["vocab_size"]))
        prev_hidden = np.zeros((m,hidden_nodes))
        prev_cell = np.zeros((m,hidden_nodes))

        self.errors = dict(input_gate=input_gate, forget_gate=forget_gate, gate_gate=gate_gate, output_gate=output_gate, cell=cell, hidden=hidden, output=output, prev_cell=prev_cell, prev_hidden=prev_hidden)



    def forward(self, X, prev_hidden, prev_cell, time_step, weights):
        input_gate = self.inputGate(X,weights['Whi'],weights['Uii'],weights['bhi'],weights['bii'],prev_hidden,time_step,self.sigmoid)
        forget_gate = self.forgetGate(X,weights['Whf'],weights['Uif'],weights['bhf'],weights['bif'],prev_hidden,time_step,self.sigmoid)
        output_gate = self.outputGate(X,weights['Who'],weights['Uio'],weights['bho'],weights['bio'],prev_hidden,time_step,self.sigmoid)
        gate_gate = self.gateGate(X,weights['Whg'],weights['Uig'],weights['bhg'],weights['big'],prev_hidden,time_step,self.tanh)
        current_cell = self.cellState(forget_gate,prev_cell,input_gate,gate_gate)
        current_hidden = output_gate * self.tanh(current_cell)
        out = weights["b"] + np.dot(current_hidden,weights["V"].T)
        output = self.softmax(out)
        self.cache = dict(input_gate=input_gate, forget_gate=forget_gate, gate_gate=gate_gate, output_gate=output_gate, current_cell=current_cell, current_hidden=current_hidden, output=output, prev_cell=prev_cell, prev_hidden=prev_hidden)

    @staticmethod
    def inputGate(X, W, U, bh, bi, prev_hidden, t, sig):
        from_hidden = np.dot(prev_hidden, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return sig(from_hidden + from_input)



    @staticmethod
    def forgetGate(X, W, U, bh, bi, prev_hidden, t, sig):
        from_hidden = np.dot(prev_hidden, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return sig(from_hidden + from_input)


    @staticmethod
    def outputGate(X, W, U, bh, bi, prev_hidden, t, sig):
        from_hidden = np.dot(prev_hidden, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return sig(from_hidden + from_input)

    @staticmethod
    def gateGate(X, W, U, bh, bi, prev_hidden, t, tanh):
        from_hidden = np.dot(prev_hidden, W.T) + bh
        from_input = (U[:,X[:,t]]).T + bi
        return tanh(from_hidden + from_input)

    @staticmethod
    def cellState(forget_gate, prev_cell, input_gate, gate_gate):
        return forget_gate * prev_cell + input_gate * gate_gate


    @staticmethod
    def sigmoid(z):
        #receives m X hidden_nodes
        return 1.0/(1 + np.exp(-z))

    @staticmethod
    def dsigmoid(a):
        return a * (1-a)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def dtanh(a):
        return 1 - a ** 2

    @staticmethod
    def softmax(outputs):
        l = len(outputs)
        outputs -= np.max(outputs,axis=-1).reshape(l,1) #for numeric stability
        expo = np.exp(outputs)
        return 1.0*expo/np.sum(expo,axis=-1).reshape(l,1)


    def backprop(self):
        pass
