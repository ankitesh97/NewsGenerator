

import numpy as np
import pickle
from preprocessor import preprocess
from GruCell import GRUCell
import json
from joblib import Parallel, delayed
import time
import multiprocessing
import sys

p_file = open('params.json','r')
p = json.loads(p_file.read())
params = p["gru"]

start = time.time()
title_len = 12

MODEL_FILE = 'modelGruv1'
train_size = params['training_size']
gradCheck = False


np.random.seed(2)


def execParallel(self,X,y,index):
    y_predicted, cells = self.forwardProp(X)
    J = self.softmaxLoss(y_predicted, y)
    dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb = self.backprop(X,y,cells)
    return index, J, dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb

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
        self.losses = []
        self.losses_after_epochs = []
        self.momentum1 = []
        self.momentum2 = []
        self.alpha = params['alpha']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.offset = params['offset']
        self.update_count = 0
        self.batch_size = params["batch_size"]["val"]
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
        weights = [self.Whz, self.Whr, self.Wg, self.Uiz ,self.Uir,self.Ug,self.bhz,self.bhr,self.bg,self.biz,self.bir,self.big,self.V,self.b]

        for i in range(len(weights)):
            self.momentum1.append(np.zeros(weights[i].shape))
            self.momentum2.append(np.zeros(weights[i].shape))

    def get_weights(self):
        weights = dict(Whz=self.Whz, Whr=self.Whr, Wg=self.Wg, Uiz=self.Uiz, Uir=self.Uir, Ug=self.Ug, bhz=self.bhz, bhr=self.bhr, bg=self.bg, biz=self.biz, bir=self.bir, big=self.big, V=self.V, b=self.b)
        return weights


    def train(self):
        obj = preprocess()
        data = obj.load()
        X = np.array(list(data.X_train[:])).astype(int)
        y = np.array(list(data.y_train[:])).astype(int)
        if train_size != -1:
            X = np.array(list(data.X_train[:train_size]))
            y = np.array(list(data.y_train[:train_size]))

        print "Everything loaded starting training"
        sys.stdout.flush()
        if gradCheck:
            self.gradientCheckTrue(X,y)
        else:
            self.miniBatchGd(X,y,data.word_to_index,data.index_to_word)


    def forwardProp(self, X):
        cells = []
        m, T = X.shape
        prev_hidden = np.ones((m,self.hidden_nodes))
        weights = self.get_weights()
        predicted = np.zeros((m,T,self.word_dim))
        for t in range(T):
            cellt = GRUCell(m)
            cellt.forward(X, prev_hidden, t, weights)
            predicted[:,t] = cellt.cache['output']
            prev_hidden = cellt.cache['current_hidden']
            cells.append(cellt)

        return predicted, cells


    def backprop(self, X, y, cells):
        m, T = X.shape
        error_from_next_cell = np.zeros((m,self.hidden_nodes))
        weights = self.get_weights()
        dJdWhz = np.zeros(self.Whz.shape)
        dJdWhr = np.zeros(self.Whr.shape)
        dJdWg = np.zeros(self.Wg.shape)
        dJdUiz = np.zeros(self.Uiz.shape)
        dJdUir = np.zeros(self.Uir.shape)
        dJdUg = np.zeros(self.Ug.shape)
        dJdbhz = np.zeros(self.bhz.shape)
        dJdbhr = np.zeros(self.bhr.shape)
        dJdbg = np.zeros(self.bg.shape)
        dJdbiz = np.zeros(self.biz.shape)
        dJdbir = np.zeros(self.bir.shape)
        dJdbig = np.zeros(self.big.shape)
        dJdV = np.zeros(self.V.shape)
        dJdb = np.zeros(self.b.shape)
        for t in range(T-1,-1,-1):
            cells[t].addErrorFromNextCell(error_from_next_cell)
            cells[t].backprop(X, y, t, weights)
            grads = cells[t].getdJdW(X, weights, t)
            error_from_next_cell = cells[t].errors['prev']
            dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb = self.unpackGrads(grads, dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb)

        return dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb

    def softmaxLoss(self, y_predicted, y):
        m = y.shape[0]
        tmp = np.array(list(np.arange(y_predicted.shape[-2]))*m)
        correct_words = y_predicted[np.arange(m).reshape(m,1), tmp.reshape(m,y_predicted.shape[-2]), y]
        correct_words[correct_words <= 1e-10] += 1e-10 #to avoid nan
        total_error = -1.0*np.log(correct_words)
        J = np.sum(total_error)
        return J


    def trainParallel(self,X,y,flag, num_cores, pool_size):
        #X here will be a mini batch this can be parallelized in the main function

        J = 0
        ite = [delayed(execParallel)(self,X[im:im+pool_size],y[im:im+pool_size], im) for im in range(0,len(X),pool_size)]
        all_return_values = Parallel(n_jobs=num_cores)(ite)
        all_return_values.sort(key=lambda j: j[0])
        dJdWhz = np.zeros(self.Whz.shape)
        dJdWhr = np.zeros(self.Whr.shape)
        dJdWg = np.zeros(self.Wg.shape)
        dJdUiz = np.zeros(self.Uiz.shape)
        dJdUir = np.zeros(self.Uir.shape)
        dJdUg = np.zeros(self.Ug.shape)
        dJdbhz = np.zeros(self.bhz.shape)
        dJdbhr = np.zeros(self.bhr.shape)
        dJdbg = np.zeros(self.bg.shape)
        dJdbiz = np.zeros(self.biz.shape)
        dJdbir = np.zeros(self.bir.shape)
        dJdbig = np.zeros(self.big.shape)
        dJdV = np.zeros(self.V.shape)
        dJdb = np.zeros(self.b.shape)

        for return_vals in all_return_values:
            im = return_vals[0]
            J += return_vals[1]
            dJdWhz += return_vals[2]
            dJdWhr += return_vals[3]
            dJdWg += return_vals[4]
            dJdUiz += return_vals[5]
            dJdUir += return_vals[6]
            dJdUg += return_vals[7]
            dJdbhz += return_vals[8]
            dJdbhr += return_vals[9]
            dJdbg += return_vals[10]
            dJdbiz += return_vals[11]
            dJdbir += return_vals[12]
            dJdbig += return_vals[13]
            dJdV += return_vals[14]
            dJdb += return_vals[15]



        self.losses.append(J)
        return dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb


    @staticmethod
    def unpackGrads(gradients,dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb):

        dJdWhz += gradients['dJdWhz']
        dJdWhr += gradients['dJdWhr']
        dJdWg += gradients['dJdWg']
        dJdUiz += gradients['dJdUiz']
        dJdUir += gradients['dJdUir']
        dJdUg += gradients['dJdUg']
        dJdbhz += gradients['dJdbhz']
        dJdbhr += gradients['dJdbhr']
        dJdbg += gradients['dJdbg']
        dJdbiz += gradients['dJdbiz']
        dJdbir += gradients['dJdbir']
        dJdbig += gradients['dJdbig']
        dJdV += gradients['dJdV']
        dJdb += gradients['dJdb']
        return dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb


    def updateParamsAdam(self,dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb, n_iteration):

        t = n_iteration
        grads = [dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb]

        for i in range(len(grads)):
            self.momentum1[i] = self.beta1*self.momentum1[i] + (1 - self.beta1) * grads[i]

        for i in range(len(grads)):
            self.momentum2[i] = self.beta2*self.momentum2[i] + (1 - self.beta2) * (grads[i]**2)


        mu1 = [0 for i in range(len(grads))]
        mu2 = [0 for i in range(len(grads))]

        for i in range(len(grads)):
            mu1[i] = 1.0 * self.momentum1[i]/(1 - self.beta1**t)
            mu2[i] = 1.0 * self.momentum2[i]/(1 - self.beta2**t)

        weigths = [self.Whz, self.Whr, self.Wg, self.Uiz ,self.Uir,self.Ug,self.bhz,self.bhr,self.bg,self.biz,self.bir,self.big,self.V,self.b]

        for i in range(len(weigths)):
            weigths[i] -= self.alpha * (mu1[i]/np.sqrt(mu2[i]+self.offset))

        self.Whz, self.Whr, self.Wg, self.Uiz ,self.Uir,self.Ug,self.bhz,self.bhr,self.bg,self.biz,self.bir,self.big,self.V,self.b = weigths

    def predict(self,X):
        output, _ = self.forwardProp(X)
        return output

    def generateSent(self, word_to_index, count,index_to_word):
        start_index = word_to_index['SENTENCE_START']
        end_index = word_to_index['SENTENCE_END']
        unknown = word_to_index['UNKNOWN_TOKEN']
        all_sent = []
        #generate 5 sentences
        for i in range(count):
            new_sent = [[start_index]]
            while new_sent[0][-1] != end_index and len(new_sent[0])<=title_len:

                s = np.array(new_sent)
                next_word_probabs = self.predict(s)[-1][-1]
                sampled_word = unknown
                while sampled_word == unknown:
                    samples = np.random.multinomial(1,next_word_probabs) #sample some random word
                    sampled_word = np.argmax(samples)
                new_sent[-1].append(sampled_word)

            if new_sent[-1][-1] == end_index:
                new_sent[-1].pop()

            s = ' '.join([index_to_word[x] for x in new_sent[-1][1:]])

            all_sent.append(s)


        return all_sent

    def miniBatchGd(self,X,y,word_to_index,index_to_word):
        n_epochs = params['epochs']
        zipped = zip(X,y)
        num_cores = 0
        pool_size = 0
        J = -1
        count = 0
        m = X.shape[0]

        parallel_flag = params["process_parallel"]
        if parallel_flag == "True":
            parallel_flag = True
        else:
            parallel_flag = False
        if(parallel_flag):
            num_cores = multiprocessing.cpu_count()
            pool_size = self.batch_size/num_cores
        for epochs in xrange(n_epochs):
            if(epochs%3==0):
                #forward propogate and get the loss
                output, _ = self.forwardProp(X)
                L = 1.0 * self.softmaxLoss(output, y)/m
                print "Epoch: "+str(epochs)+" over all Loss: "+str(L)+" time: "+str(time.time()-start)
                sys.stdout.flush()
                self.losses_after_epochs.append(L)

            if(epochs%5==0):
                print "-------------------------------------"
                print "Sentences at Epoch: "+str(epochs)
                try:
                    for num, x in enumerate(self.generateSent(word_to_index, 5,index_to_word)):
                        print str(num+1)+' --- '+x

                except Exception as e:
                    print "some unicode charachter occured"
                print "-------------------------------------"
                sys.stdout.flush()

                with open("controlTraining.txt",'r') as f:
                    control = f.read()
                    if control.strip() == "1":
                       print "stopping the training process .........."
                       sys.stdout.flush()
                       break


            np.random.shuffle(zipped)
            X,y = zip(*zipped)
            X = np.array(X)
            y = np.array(y)
            for i in xrange(0,X.shape[0],self.batch_size):
                #get the current mini batch
                X_mini = X[i:i+self.batch_size]
                y_mini = y[i:i+self.batch_size]
                if parallel_flag:
                    count += 1
                    dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb = self.trainParallel(X_mini, y_mini, parallel_flag, num_cores, pool_size)
                    self.updateParamsAdam(dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb, count)


            #decay the learning rate
            self.alpha = 1.0*self.alpha/(1+epochs)

        prev_hidden = np.zeros((X.shape[0],self.hidden_nodes))
        output, _ = self.forwardProp(X)
        L = self.softmaxLoss(output, y)
        print "Epoch: "+str(epochs)+" over all Loss after training: "+str(L)+" time: "+str(time.time()-start)
        sys.stdout.flush()
        self.losses_after_epochs.append(L)
        sys.stdout.flush()

    def gradientCheckTrue(self,X,y):
       epsi = 1e-7
       X = X[:,:3]
       y = y[:,:3]
       y_predicted,cells = self.forwardProp(X)
       dJdWhz, dJdWhr, dJdWg, dJdUiz, dJdUir, dJdUg, dJdbhz, dJdbhr, dJdbg, dJdbiz, dJdbir, dJdbig, dJdV, dJdb = self.backprop(X,y,cells)
       approx = np.zeros(self.Ug.shape)
       approxb = np.zeros(self.big.shape)


       #check u

    #    for bias
       for i in range(self.big.shape[0]):
           self.big[i] += epsi
           out, _ = self.forwardProp(X)
           J1 = self.softmaxLoss(out, y)
           self.big[i] -= 2*epsi
           out, _ = self.forwardProp(X)
           J2 = self.softmaxLoss(out, y)
           approxb[i] = (1.0*(J1-J2))/(2*epsi)
           self.big[i] += epsi

       print dJdbig
       print approxb
       nume = np.linalg.norm(approxb-dJdbig)
       deno = np.linalg.norm(dJdbig) + np.linalg.norm(approxb)
       print "ratio is " +  str(nume/deno)

       #
       for i in range(self.Ug.shape[0]):
           for j in range(self.Ug.shape[-1]):
            #    print i, j
               self.Ug[i][j] += epsi
               out, _ = self.forwardProp(X)
               J1 = self.softmaxLoss(out, y)
               self.Ug[i][j] -= 2*epsi
               out, _ = self.forwardProp(X)
               J2 = self.softmaxLoss(out, y)
               approx[i][j] = (1.0*(J1-J2))/(2*epsi)
               self.Ug[i][j] += epsi

       print dJdUg
       print approx
       nume = np.linalg.norm(approx-dJdUg)
       deno = np.linalg.norm(dJdUg) + np.linalg.norm(approx)
       print "ratio is " +  str(nume/deno)
       #

if __name__ == '__main__':
    model = Gru()
    model.train()
    pickle_file_sampled_data = open('pickledfiles/'+MODEL_FILE,'w')
    pickle.dump(model, pickle_file_sampled_data)
    pickle_file_sampled_data.close()
    print("--- Training completed in seconds %s---" % (time.time() - start))
