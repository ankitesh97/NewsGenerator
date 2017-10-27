
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
        current_cell_tan = self.tanh(current_cell)
        current_hidden = output_gate * current_cell_tan
        out = weights["b"] + np.dot(current_hidden,weights["V"].T)
        output = self.softmax(out)
        self.cache = dict(input_gate=input_gate, forget_gate=forget_gate, gate_gate=gate_gate, output_gate=output_gate, current_cell=current_cell, current_cell_tan=current_cell_tan,  current_hidden=current_hidden, output=output, prev_cell=prev_cell, prev_hidden=prev_hidden)

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


    def backprop(self, X, y, t, weights):
        #get the errors in the ouput layer
        dy = self.cache['output']
        dy[np.arange(X.shape[0]),y[:,t]] -= 1
        #pass to the hidden layer
        dhidden = self.errors['hidden']
        dhidden += np.dot(dy, weights['V']) #this is the error propogated to both the components that added up to get the current hidden state

        #error at output gate
        doutput_gate = (dhidden * self.cache['current_cell_tan']) * self.dsigmoid(self.cache['output_gate'])


        #error at cell gate
        dcell = self.errors['cell']
        dcell_from_ht = (dhidden * self.cache['output_gate']) * self.dtanh(self.cache['current_cell_tan'])
        dcell += dcell_from_ht

        #
        dprev_cell = dcell * self.cache['forget_gate']
        dforget_gate = (dcell * self.cache['prev_cell']) * self.dsigmoid(self.cache['forget_gate'])
        dinput_gate = (dcell * self.cache['gate_gate']) * self.dsigmoid(self.cache['input_gate'])
        dgate_gate = (dcell * self.cache['input_gate']) * self.dtanh(self.cache['gate_gate'])

        #dprev_hidden
        dprev_hidden = np.dot(dgate_gate, weights['Whg'])
        dprev_hidden += np.dot(doutput_gate, weights['Who'])
        dprev_hidden += np.dot(dforget_gate, weights['Whf'])
        dprev_hidden += np.dot(dinput_gate, weights['Whi'])


        errors = dict(input_gate=dinput_gate, forget_gate=dforget_gate, gate_gate=dgate_gate, output_gate=doutput_gate, cell=dcell, hidden=dhidden, output=dy, prev_cell=dprev_cell, prev_hidden=dprev_hidden)
        self.errors = errors

    def addErrorFromNextCell(self, hidden, cell):
        self.errors['cell'] += cell
        self.errors['hidden'] += hidden


    @staticmethod
    def _calcForinputU(shape, error, X, t):
        dJdU = np.zeros(shape)
        if len(set(X[:,t])) == len(X[:,t]):
            dJdU[:,X[:,t]] += error.T
        else:
            update_cols = X[:,t]
            tpose = error.T
            for dps in range(len(update_cols)):
                dJdU[:,update_cols[dps]] += tpose[:,dps]

        return dJdU

    # order ['Whi','Whf','Who','Whg','Uii','Uif','Uio','Uig','V','bhi','bhf','bho','bhg','bii','bif','bio','big','b']
    def getdJdW(self, X, weights, t ):

        ts = self.cache['current_hidden'].shape
        dJdV = np.matmul(self.errors['output'].reshape(self.errors['output'].shape+(1,)), self.cache['current_hidden'].reshape((ts[0],1,ts[1])))
        dJdV = np.sum(dJdV,axis=0)
        dJdb = np.sum(self.errors['output'],axis=0)

        ths = self.cache['prev_hidden'].shape

        ts = self.errors['input_gate'].shape
        dJdWhi = np.matmul(self.errors['input_gate'].reshape(ts+(1,)), self.cache['prev_hidden'].reshape((ths[0],1,ths[-1])))
        dJdWhi = np.sum(dJdWhi, axis=0)
        dJdUii = self._calcForinputU(weights['Uii'].shape, self.errors['input_gate'], X, t)
        dJdbhi = np.sum(self.errors['input_gate'], axis = 0)
        dJdbii = np.sum(self.errors['input_gate'], axis = 0)


        ts = self.errors['forget_gate'].shape
        dJdWhf = np.matmul(self.errors['forget_gate'].reshape(ts+(1,)), self.cache['prev_hidden'].reshape((ths[0],1,ths[-1])))
        dJdWhf = np.sum(dJdWhf, axis=0)
        dJdUif = self._calcForinputU(weights['Uif'].shape, self.errors['forget_gate'], X, t)
        dJdbhf = np.sum(self.errors['forget_gate'], axis = 0)
        dJdbif = np.sum(self.errors['forget_gate'], axis = 0)


        ts = self.errors['output_gate'].shape
        dJdWho = np.matmul(self.errors['output_gate'].reshape(ts+(1,)), self.cache['prev_hidden'].reshape((ths[0],1,ths[-1])))
        dJdWho = np.sum(dJdWho, axis=0)
        dJdUio = self._calcForinputU(weights['Uio'].shape, self.errors['output_gate'], X, t)
        dJdbho = np.sum(self.errors['output_gate'], axis = 0)
        dJdbio = np.sum(self.errors['output_gate'], axis = 0)



        ts = self.errors['gate_gate'].shape
        dJdWhg = np.matmul(self.errors['gate_gate'].reshape(ts+(1,)), self.cache['prev_hidden'].reshape((ths[0],1,ths[-1])))
        dJdWhg = np.sum(dJdWhg, axis=0)
        dJdUig = self._calcForinputU(weights['Uig'].shape, self.errors['gate_gate'], X, t)
        dJdbhg = np.sum(self.errors['gate_gate'], axis = 0)
        dJdbig = np.sum(self.errors['gate_gate'], axis = 0)

        grads = [dJdWhi,dJdWhf,dJdWho,dJdWhg,dJdUii,dJdUif,dJdUio,dJdUig,dJdV,dJdbhi,dJdbhf,dJdbho,dJdbhg,dJdbii,dJdbif,dJdbio,dJdbig,dJdb]

        return grads
