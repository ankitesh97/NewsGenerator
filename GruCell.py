
import numpy as np
import pickle
from preprocessor import preprocess
import json
from joblib import Parallel, delayed
import time
import multiprocessing
import sys
from Gru import Gru

p_file = open('params.json','r')
p = json.loads(p_file.read())
params = p["gru"]

class GRUCell:

    #update gate(z), reset gate(r), memory gate(g)
    def __init__(self,m):
        self.cache = None
        self.errors = None  # this will be initialized
        self.initErrors(m)




    def initErrors(self,m):
        hidden_nodes = params["hidden_nodes"]
        update_gate = np.zeros((m,hidden_nodes))
        reset_gate = np.zeros((m,hidden_nodes))
        memory_gate = np.zeros((m,hidden_nodes))
        hidden = np.zeros((m,hidden_nodes))
        output = np.zeros((m,p["preprocess"]["vocab_size"]))
        prev = np.zeros((m,hidden_nodes))
        self.errors = dict(update_gate=update_gate, reset_gate=reset_gate, memory_gate=memory_gate, hidden=hidden, output=output, prev=prev)

    # X is the of shape m X time_steps
    # prev_hidden m X hidden_size
    def forward(self, X, prev_hidden, time_step, weights):
        update_gate = self.updateGate(X, weights["Whz"], weights["Uiz"], weights["bhz"], weights["biz"], prev_hidden, time_step, self.sigmoid)
        reset_gate = self.resetGate(X, weights["Whr"], weights["Uir"], weights["bhr"], weights["bir"], prev_hidden, time_step, self.sigmoid)
        memory_gate = self.memoryGate(X, weights["Wg"], weights["Ug"], weights["bg"], weights["big"], reset_gate, prev_hidden, time_step, self.tanh)
        current_hidden = update_gate * memory_gate + (1-update_gate) * prev_hidden
        out = weights["b"] + np.dot(current_hidden,weights["V"].T)
        output = self.softmax(out)
        self.cache = dict(update_gate=update_gate, reset_gate = reset_gate, memory_gate = memory_gate, current_hidden = current_hidden, output=output, prev_hidden=prev_hidden)

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
        print outputs.shape
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

        #error at the update gate
        dzt = (self.cache['memory_gate'] * dhidden) #this is from the 1st component of ht
        dzt += (-1) * (self.cache['prev_hidden'] * dhidden) #from the 2nd component of ht
        dzt = dzt * self.dsigmoid(self.cache['update_gate']) #this is to pass through the sigmoid gate

        #error at the memory gate
        dgt = self.cache['update_gate'] * dhidden * self.dtanh(self.cache['memory_gate'])

        #error from memory gate to its components
        dintert = np.dot(dgt, weights['Wg'] )

        #error at reset gate
        drt = (dintert * self.cache['prev_hidden']) * self.dsigmoid(self.cache['reset_gate'])

        #error to pass to previous cell
        #from the curr_hidden
        dprev = (1 - self.cache['update_gate']) * dhidden
        # from the memory gate
        dprev += self.cache['reset_gate'] * dintert
        #from update gate
        dprev += np.dot(dzt, weights['Whz'])
        #from reset gate
        dprev += np.dot(drt, weights['Whr'])

        self.errors['output'] = dy
        self.errors['update_gate'] = dzt
        self.errors['reset_gate'] = drt
        self.errors['hidden'] = dhidden
        self.errors['memory_gate'] = dgt
        self.errors['prev'] = dprev


    def addErrorFromNextCell(self,error_from_next):
        self.errors['hidden'] += error_from_next

    def getdJdW(self,X,weights,t):
        ts = self.cache['current_hidden'].shape
        dJdV = np.matmul(self.errors['output'].reshape(self.errors['output'].shape+(1,)), self.cache['current_hidden'].reshape((ts[0],1,ts[1])))
        dJdV = np.sum(dJdV,axis=0)

        dJdb = np.sum(self.errors['output'],axis=0)

        #for parameters of the variable inter in the memory gate
        ain_inter = self.cache['reset_gate'] * self.cache['prev_hidden']
        ain_inter = ain_inter.reshape((ain_inter.shape[0],1,ain_inter.shape[1]))
        dJdWg = np.matmul(self.errors['memory_gate'].reshape(self.errors['memory_gate'].shape+(1,)), ain_inter)
        dJdWg = np.sum(dJdWg,axis=0)

        dJdUg = np.zeros(weights['Ug'].shape)
        if len(set(X[:,t])) == len(X[:,t]):
            dJdUg[:,X[:,t]] += self.errors['memory_gate'].T
        else:
            update_cols = X[:,t]
            tpose = self.errors['memory_gate'].T
            for dps in range(len(update_cols)):
                dJdUg[:,update_cols[dps]] += tpose[:,dps]

        dJdbg = np.sum(self.errors['memory_gate'], axis=0)
        dJdbig = np.sum(self.errors['memory_gate'], axis=0)


        #weights in update gate
        ts = self.errors['update_gate'].shape
        ths = self.cache['prev_hidden'].shape
        dJdWhz = np.matmul(self.errors['update_gate'].reshape(ts+(1,)), self.cache['prev_hidden'].reshape((ths[0],1,ths[-1])))
        dJdWhz = np.sum(dJdWhz, axis=0)
        dJdbhz = np.sum(self.errors['update_gate'],axis=0)
        dJdbiz = np.sum(self.errors['update_gate'],axis=0)

        dJdUiz = np.zeros(weights['Uiz'].shape)
        if len(set(X[:,t])) == len(X[:,t]):
            dJdUiz[:,X[:,t]] += self.errors['update_gate'].T
        else:
            update_cols = X[:,t]
            tpose = self.errors['update_gate'].T
            for dps in range(len(update_cols)):
                dJdUiz[:,update_cols[dps]] += tpose[:,dps]


        #weights in reset gate
        dJdWhr = np.matmul(self.errors['reset_gate'].reshape(ts+(1,)), self.cache['prev_hidden'].reshape((ths[0],1,ths[-1])))
        dJdWhr = np.sum(dJdWhr, axis=0)
        dJdbhr = np.sum(self.errors['reset_gate'],axis=0)
        dJdbir = np.sum(self.errors['reset_gate'],axis=0)

        dJdUir = np.zeros(weights['Uir'].shape)
        if len(set(X[:,t])) == len(X[:,t]):
            dJdUir[:,X[:,t]] += self.errors['reset_gate'].T
        else:
            update_cols = X[:,t]
            tpose = self.errors['reset_gate'].T
            for dps in range(len(update_cols)):
                dJdUir[:,update_cols[dps]] += tpose[:,dps]


        dJdWeights = dict(dJdWhz=dJdWhz, dJdWhr=dJdWhr, dJdWg=dJdWg, dJdUiz=dJdUiz, dJdUir=dJdUir, dJdUg=dJdUg, dJdbhz=dJdbhz, dJdbhr=dJdbhr, dJdbg=dJdbg, dJdbiz=dJdbiz, dJdbir=dJdbir, dJdbig=dJdbig, dJdV=dJdV, dJdb=dJdb)
        return dJdWeights


if __name__ == '__main__':
    obj = preprocess()
    data = obj.load()
    gru  = Gru()
    weights = gru.get_weights()
    X = np.array(list(data.X_train[:])).astype(int)[:15]
    y = np.array(list(data.y_train[:])).astype(int)[:15]
    cell = GRUCell(X.shape[0])
    prev_hidden = np.zeros((X.shape[0],params["hidden_nodes"]))
    cell.forward(X, prev_hidden, 0, weights)
    # print cell.cache
    cell.backprop(X,y,0,weights)
    # print cell.errors
    dj = cell.getdJdW(X,weights,0)
    print dj
