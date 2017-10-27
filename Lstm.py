

import numpy as np
import pickle
from preprocessor import preprocess
from LstmCell import LstmCell
import json
from joblib import Parallel, delayed
import time
import multiprocessing
import sys

p_file = open('params.json','r')
p = json.loads(p_file.read())
params = p["lstm"]

start = time.time()
title_len = 12

MODEL_FILE = 'modelLstmv1'
train_size = params['training_size']
gradCheck = True


np.random.seed(2)


def execParallel(self,X,y,index):
    y_predicted, cells = self.forwardProp(X)
    J = self.softmaxLoss(y_predicted, y)
    grads = self.backprop(X,y,cells)
    return index, J, grads


class Lstm:
    def __init__(self):

        self.weights = []
        # order ['Whi','Whf','Who','Whg','Uii','Uif','Uio','Uig','V','bhi','bhf','bho','bhg','bii','bif','bio','big','b']
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
        Whi = np.random.randn(self.gate_params_shape[0],self.gate_params_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))
        Whf = np.random.randn(self.gate_params_shape[0],self.gate_params_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))
        Who = np.random.randn(self.gate_params_shape[0],self.gate_params_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))
        Whg = np.random.randn(self.gate_params_shape[0],self.gate_params_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))

        Uii = np.random.randn(self.gate_params_shape[1],self.word_dim) * np.sqrt(1.0/(1+self.word_dim))
        Uif = np.random.randn(self.gate_params_shape[1],self.word_dim) * np.sqrt(1.0/(1+self.word_dim))
        Uio = np.random.randn(self.gate_params_shape[1],self.word_dim) * np.sqrt(1.0/(1+self.word_dim))
        Uig = np.random.randn(self.gate_params_shape[1],self.word_dim) * np.sqrt(1.0/(1+self.word_dim))

        bhi = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes))
        bhf = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes))
        bho = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes))
        bhg = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes))

        bii = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim))
        bif = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim))
        bio = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim))
        big = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim))

        V = np.random.randn(self.word_dim,self.gate_params_shape[1]) * np.sqrt(1.0/self.hidden_nodes+1)
        b = np.random.randn(self.word_dim) * np.sqrt(1.0/(1+self.word_dim))

        self.weights = [Whi,Whf,Who,Whg,Uii,Uif,Uio,Uig,V,bhi,bhf,bho,bhg,bii,bif,bio,big,b]

        for i in range(len(self.weights)):
            self.momentum1.append(np.zeros(self.weights[i].shape))
            self.momentum2.append(np.zeros(self.weights[i].shape))

    def get_weights(self):
        names = ['Whi','Whf','Who','Whg','Uii','Uif','Uio','Uig','V','bhi','bhf','bho','bhg','bii','bif','bio','big','b']
        w = {}
        for i in range(len(self.weights)):
            w[names[i]] = self.weights[i]
        return w


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
        prev_hidden = np.zeros((m,self.hidden_nodes))
        prev_cell = np.zeros((m,self.hidden_nodes))
        w = self.get_weights()
        predicted = np.zeros((m,T,self.word_dim))
        for t in range(T):
            cellt = LstmCell(m)
            cellt.forward(X, prev_hidden, prev_cell, t, w)
            predicted[:,t] = cellt.cache['output']
            prev_hidden = cellt.cache['current_hidden']
            prev_cell = cellt.cache['current_cell']
            cells.append(cellt)

        return predicted, cells


    def backprop(self, X, y, cells):
        m, T = X.shape
        error_from_next_cell = np.zeros((m,self.hidden_nodes))
        cell_t_from_next_cell = np.zeros((m,self.hidden_nodes))
        weights = self.get_weights()
        grads = []
        for t in range(T-1,-1,-1):
            cells[t].addErrorFromNextCell(error_from_next_cell, cell_t_from_next_cell)
            cells[t].backprop(X, y, t, weights)
            grads_current = cells[t].getdJdW(X, weights, t)
            error_from_next_cell = cells[t].errors['prev_hidden']
            cell_t_from_next_cell = cells[t].errors['prev_cell']
            if t != T-1:
                grads = self.unpackGrads(grads, grads_current)
            else:
                grads = grads_current


        return grads


    @staticmethod
    def unpackGrads(grads, grads_current):

        for i in range(len(grads)):
            grads[i] += grads_current[i]
        return grads

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
        grads = []
        for return_vals in all_return_values:
            im = return_vals[0]
            J += return_vals[1]
            grads_curr = return_vals[2]
            if(len(grads)==0):
                grads = grads_curr
            else:
                for i in range(len(grads)):
                    grads[i] += grads_curr[i]


        self.losses.append(J)
        return grads


    def updateParamsAdam(self,grads, n_iteration):

        t = n_iteration

        for i in range(len(grads)):
            self.momentum1[i] = self.beta1*self.momentum1[i] + (1 - self.beta1) * grads[i]

        for i in range(len(grads)):
            self.momentum2[i] = self.beta2*self.momentum2[i] + (1 - self.beta2) * (grads[i]**2)


        mu1 = [0 for i in range(len(grads))]
        mu2 = [0 for i in range(len(grads))]

        for i in range(len(grads)):
            mu1[i] = 1.0 * self.momentum1[i]/(1 - self.beta1**t)
            mu2[i] = 1.0 * self.momentum2[i]/(1 - self.beta2**t)


        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * (mu1[i]/np.sqrt(mu2[i]+self.offset))




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
                output, _ = self.forwardProp(X[:3000])
                L = 1.0 * self.softmaxLoss(output, y[:3000])/3000
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
                    grads = self.trainParallel(X_mini, y_mini, parallel_flag, num_cores, pool_size)
                    self.updateParamsAdam(grads, count)


            #decay the learning rate
            self.alpha = 1.0*self.alpha/(1+epochs)

        prev_hidden = np.zeros((X.shape[0],self.hidden_nodes))
        output, _ = self.forwardProp(X[:3000])
        L = self.softmaxLoss(output, y[:3000])
        print "Epoch: "+str(epochs)+" over all Loss after training: "+str(L)+" time: "+str(time.time()-start)
        sys.stdout.flush()
        self.losses_after_epochs.append(L)
        sys.stdout.flush()

    def gradientCheckTrue(self,X,y):
       epsi = 1e-7
       X = X[:,:2]
       y = y[:,:2]
       y_predicted,cells = self.forwardProp(X)
       grads = self.backprop(X,y,cells)
       names = ['Whi','Whf','Who','Whg','Uii','Uif','Uio','Uig','V','bhi','bhf','bho','bhg','bii','bif','bio','big','b']

       for i in  range(len(self.weights)):
           approx = np.zeros(self.weights[i].shape)

           if len(self.weights[i].shape) > 1:
               for r in range(self.weights[i].shape[0]):
                   for c in range(self.weights[i].shape[1]):
                       self.weights[i][r][c] += epsi
                       out, _  = self.forwardProp(X)
                       J1 = self.softmaxLoss(out, y)
                       self.weights[i][r][c] -= 2*epsi
                       out, _  = self.forwardProp(X)
                       J2 = self.softmaxLoss(out, y)
                       approx[r][c] = (1.0*(J1-J2))/(2*epsi)
                       self.weights[i][r][c] += epsi

               nume = np.linalg.norm(approx-grads[i])
               deno = np.linalg.norm(grads[i]) + np.linalg.norm(approx)
               print "ratio of "+names[i]+" " +  str(nume/deno)
           else:
               for j in range(len(self.weights[i])):
                   self.weights[i][j] += epsi
                   out, _  = self.forwardProp(X)
                   J1 = self.softmaxLoss(out, y)
                   self.weights[i][j] -= 2*epsi
                   out, _  = self.forwardProp(X)
                   J2 = self.softmaxLoss(out, y)
                   approx[j] = (1.0*(J1-J2))/(2*epsi)
                   self.weights[i][j] += epsi

            #    print approx
            #    print grads[i]
               nume = np.linalg.norm(approx-grads[i])
               deno = np.linalg.norm(grads[i]) + np.linalg.norm(approx)
               print "ratio of "+names[i]+" " +  str(nume/deno)


if __name__ == '__main__':
    model = Lstm()
    model.train()
    pickle_file_sampled_data = open('pickledfiles/'+MODEL_FILE,'w')
    pickle.dump(model, pickle_file_sampled_data)
    pickle_file_sampled_data.close()
    print("--- Training completed in seconds %s---" % (time.time() - start))
