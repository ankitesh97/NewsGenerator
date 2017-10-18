import numpy as np
import pickle
from preprocessor import preprocess
import json
from joblib import Parallel, delayed
import time
import multiprocessing
import sys

p_file = open('params.json','r')
params = json.loads(p_file.read())
np.random.seed(10)
start = time.time()



MODEL_FILE = 'modelv1'
train_size = params['training_size']

def execParallel(self,X,y,t,prev_hidden,im):
    y_predicted,hidden_state_info_activated = self.forwardProp(X[:,t:t+self.truncate],prev_hidden)
    tmp = hidden_state_info_activated[:,-2]
    J = self.softmaxLoss(y_predicted, y[:,t:t+self.truncate])
    #now backpropogate
    dJdV, dJdW, dJdU, dJdbho , dJdbhh, dJdbih  = self.tbptt(X[:,t:t+self.truncate], y[:,t:t+self.truncate], y_predicted,hidden_state_info_activated,(t+self.truncate-1)%self.truncate)
    return im,tmp, J, dJdV, dJdW, dJdU, dJdbho , dJdbhh, dJdbih

class RNNModel():
    def __init__(self):
        self.W = [] #weights between hidden to hidden
        self.U = [] #weights between input to hidden
        self.V = [] #weights between hidden to output'
        self.bh = 0 #bias of hidden layer
        self.bi = 0 #bias of input layer
        self.word_dim = params['preprocess']['vocab_size']
        self.vocab_size = params["preprocess"]["vocab_size"]
        self.hidden_nodes = params['hidden_nodes']
        self.w_shape = (self.hidden_nodes,self.hidden_nodes)
        self.u_shape = (self.hidden_nodes,self.word_dim)
        self.v_shape = (self.vocab_size, self.hidden_nodes)
        self.hidden_states_info  = []
        self.losses = [-2] #to keep track of the loss
        self.losses_after_epochs = []
        self.truncate = params['truncation']

        #some values for adam optimization
        self.momentum1_w = np.zeros(self.w_shape)
        self.momentum1_u = np.zeros(self.u_shape)
        self.momentum1_v = np.zeros(self.v_shape)
        self.momentum1_bhh = np.zeros(self.hidden_size)
        self.momentum1_bho = np.zeros(self.word_dim)
        self.momentum1_bih = np.zeros(self.hidden_size)

        self.momentum2_w = np.zeros(self.w_shape)
        self.momentum2_u = np.zeros(self.u_shape)
        self.momentum2_v = np.zeros(self.v_shape)
        self.momentum1_bhh = np.zeros(self.hidden_size)
        self.momentum1_bho = np.zeros(self.word_dim)
        self.momentum1_bih = np.zeros(self.hidden_size)
        #hyperparamters
        self.alpha = params['alpha']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.offset = params['offset']

        self.update_count = 0

        self.batch_size = params["batch_size"]["val"]


    def train(self):
        obj = preprocess()
        data = obj.load()
        X = np.array(list(data.X_train[:]))
        y = np.array(list(data.y_train[:]))
        if train_size != -1:
            X = np.array(list(data.X_train[:train_size]))
            y = np.array(list(data.y_train[:train_size]))
        self.randomizeParams()
        print "Everything loaded starting training"
        sys.stdout.flush()
        self.miniBatchGd(X,y)

    #takes a The whole dataset as the input and forward propogates for that
    def forwardProp(self,X,prev_hidden):
        #time steps
        # this function receives a sentence (in terms of index 2d array)
        T = X.shape[-1]
        dp = X.shape[0]

        hidden_state_info_activated = np.zeros((dp,T+1,self.hidden_nodes)) #for every training example i have a hidden state info
        hidden_state_info = np.zeros((dp,T+1,self.hidden_nodes))
        hidden_state_info[:,-1] = prev_hidden

        output = np.zeros((dp,T,self.word_dim)) #fist one should not be considered, therefore indexing is from 1 to T
        for t in np.arange(0,T):
            curr_hidden_state_info = hidden_state_info[:,t-1,:].reshape(dp,1,self.hidden_nodes)
            hidden_to_hidden = self.bhh + np.dot(curr_hidden_state_info,self.W.T).squeeze()
            input_to_hidden = self.bih + (self.U[:,X[:,t]]).T
            non_activated = hidden_to_hidden+input_to_hidden
            activated = self.tanh(non_activated)
            hidden_state_info[:,t] = non_activated
            hidden_state_info_activated[:,t] = activated
            curr_output = self.bh0 + np.dot(hidden_state_info_activated[:,t].reshape(dp,1,self.hidden_nodes),self.V.T).squeeze()
            probab = self.softmax(curr_output)
            output[:,t] = probab

        #returns the information for the whole dataset

        return output, hidden_state_info_activated


    # the predict function returns the calculated output by just calculating the max probability, x is the single sentence
    def predict(self,X):
        prev_hidden = np.zeros(self.hidden_nodes)
        output, _ = self.forwardProp(X,prev_hidden)
        return output

    def generateSent(self, word_to_index, count):
        start_index = word_to_index['SENTENCE_START']
        end_index = word_to_index['SENTENCE_END']
        unknown = word_to_index['UNKNOWN_TOKEN']
        all_sent = []
        #generate 5 sentences
        for i in range(count):
            new_sent = [[start_index]]
            while new_sent[0][-1] != end_index:
                next_word_probabs = self.predict(new_sent)[-1]
                sampled_word = unknown
                while sampled_word == unknown:
                    samples = np.random.multinomial(1,next_word_probabs) #sample some random word
                    sampled_word = np.argmax(samples)
                new_sent[-1].append(sampled_word)

            all_sent.append(new_sent[-1])


        return all_sent

    @staticmethod
    def tanh(z):
        return np.tanh(z)


    @staticmethod
    def softmax(X):
        X -= np.max(X,axis=-1).reshape(-1,1) #for numeric stability
        expo = np.exp(X)
        return 1.0*expo/np.sum(expo,axis=-1).reshape(-1,1)


    def randomizeParams(self):
        # xavier initialization
        self.W = np.random.randn(self.w_shape[0],self.w_shape[1]) * np.sqrt(1.0/(1+self.hidden_nodes))
        self.bhh = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.hidden_nodes)) #hidden to hidden layer bias
        self.U = np.random.randn(self.u_shape[0],self.u_shape[1]) * np.sqrt(1.0/(1+*self.word_dim))
        self.bih = np.random.randn(self.hidden_nodes) * np.sqrt(1.0/(1+self.word_dim)) #input to hidden bias
        self.V = np.random.randn(self.v_shape[0],self.v_shape[1]) * np.sqrt(1.0/self.hidden_nodes+1)
        self.bho = np.random.randn(self.vocab_size) * np.sqrt(1.0/(1+*self.word_dim)) #hidden to output bias




    #this is tensorflow style tbtt (apply this with minibatch)

    #takes one data point as the input and returns gradient change with respect to the weights
    def tbpttTrain(self,X,y,flag, num_cores, pool_size):
        #X here will be a mini batch this can be parallelized in the main function
        prev_hidden = np.zeros(self.hidden_nodes)
        for t in range(0,X.shape[-1],self.truncate):
            y_predicted,hidden_state_info_activated = self.forwardProp(X[:,t:t+self.truncate],prev_hidden)
            prev_hidden = hidden_state_info_activated[:][-2]
            J = self.softmaxLoss(y_predicted, y[:,t:t+self.truncate])
            self.losses.append(J)


            #now backpropogate
            dJdV, dJdW, dJdU, dJdbho, dJdbhh, dJdbih  = self.tbptt(X[:,t:t+self.truncate], y[:,t:t+self.truncate], y_predicted,hidden_state_info_activated,(t+self.truncate-1)%self.truncate)
            self.update_count += 1
            self.updateParamsAdam(dJdV, dJdW, dJdU, dJdbho, dJdbhh, dJdbih, self.update_count)
            if self.update_count % 30 == 0:
                print "After Updates: "+str(self.update_count)+" updates inside bptt Loss: " +str(J)
                sys.stdout.flush()
        print "After Updates: "+str(self.update_count)+" updates inside bptt Loss: " +str(J)
        sys.stdout.flush()


    def tbpttTrainParallel(self,X,y,flag, num_cores, pool_size):
        #X here will be a mini batch this can be parallelized in the main function
        prev_hidden = np.zeros((X.shape[0],self.hidden_nodes))

        for t in range(0,X.shape[-1],self.truncate):
            J = 0
            ite = [delayed(execParallel)(self,X[im:im+pool_size],y[im:im+pool_size],t,prev_hidden[im:im+pool_size],im) for im in range(0,len(X),pool_size)]
            all_return_values = Parallel(n_jobs=num_cores)(ite)
            all_return_values.sort(key=lambda j: j[0])
            dJdW = np.zeros(self.w_shape)
            dJdU = np.zeros(self.u_shape)
            dJdV = np.zeros(self.v_shape)
            dJdbhh = np.zeros(self.hidden_size)
            dJdbih = np.zeros(self.hidden_size)
            dJdbho = np.zeros(self.word_dim)
            for return_vals in all_return_values:
                im = return_vals[0]
                prev_hidden[im:im+pool_size] = return_vals[1]
                J += return_vals[2]
                dJdV += return_vals[3]
                dJdW += return_vals[4]
                dJdU += return_vals[5]
                dJdbho += return_vals[6]
                dJdbhh += return_vals[7]
                dJdbih += return_vals[8]


            self.update_count += 1
            self.updateParamsAdam(dJdV, dJdW, dJdU, dJdbho, dJdbhh, dJdbih, self.update_count)
            #calculate total loss here
            self.losses.append(J)
            if self.update_count % 30 == 0:
                print "After Updates: "+str(self.update_count)+" updates inside bptt Loss: " +str(J)+" time: "+str(time.time()-start)
                sys.stdout.flush()
        print "After Updates For this batch: "+str(self.update_count)+" Loss: " +str(J)
        sys.stdout.flush()



    def tbptt(self, X, y, y_predicted, hidden_state_info_activated,T):

        #error in the ouput
        m = y.shape[0]
        time_steps = y.shape[1]
        hidden_state_info_activated_actual = hidden_state_info_activated[:,:-1]
        hidden_shape = hidden_state_info_activated_actual.shape
        tmp = np.array(list(np.arange(y_predicted.shape[-2]))*m)
        dy = y_predicted
        dh =  np.zeros((m, time_steps, self.hidden_nodes))#error at hidden layer nodes shape mxTxhidden_size
        dJdW = np.zeros(self.w_shape)
        dJdU = np.zeros(self.u_shape)

        dy[np.arange(m).reshape(m,1), tmp.reshape(m,y_predicted.shape[-2]), y] -= 1
        dJdV = np.zeros(self.v_shape)
        dJdV_multi = np.matmul(dy.reshape(dy.shape[:] + (1,)), hidden_state_info_activated_actual.reshape(hidden_shape[:-1]+(1,hidden_shape[-1])))
        dJdV = np.sum(dJdV_multi,axis=(0,1))
        dJdbho = np.sum(dy,axis=(0,-2))
        dJdbih = np.zeros(self.hidden_size)
        dJdbhh = np.zeros(self.hidden_size)
        #in reverse direction
        for t in range(T, T-self.truncate-1,-1):
            dh[:,t] += np.dot(dy[:,t],self.V) * (1-hidden_state_info_activated[:,t]**2) #due to the ouput layer
            total =  np.sum(dh[:,t],axis=0)
            dJdbhh += total
            dJdbih += total
            #propogate the error back in time
            dout = dh[:,t]
            if(t != T-self.truncate):
                dh[:,t-1] += np.dot(dh[:,t],self.W) * (1-hidden_state_info_activated[:,t-1]**2)
                #now calculate the error wrt to  time weights
                ain = hidden_state_info_activated[:,t-1]
                dJdW_vect = np.matmul(dout.reshape(dout.shape[:]+(1,)), ain.reshape(ain.shape[:-1] + (1,ain.shape[-1])))
                dJdW += np.sum(dJdW_vect,axis=0)

            # grads wrt to U
            dJdU[:,X[:,t]] += dout.T


        return  dJdV, dJdW, dJdU, dJdbho, dJdbhh, dJdbih



    #this is the true tbtt apply this with SGD
    def trueTbttTrain(self):
        pass


    def trueTbtt(self):
        pass


    def softmaxLoss(self, y_predicted, y):
        m = y.shape[0]
        tmp = np.array(list(np.arange(y_predicted.shape[-2]))*m)
        correct_words = y_predicted[np.arange(m).reshape(m,1), tmp.reshape(m,y_predicted.shape[-2]), y]
        correct_words[correct_words <= 1e-10] += 1e-10 #to avoid nan
        total_error = -1.0*np.log(correct_words)
        J = np.sum(total_error)/m
        return J


    def updateParamsAdam(self,dJdV, dJdW, dJdU, dJdbho, dJdbhh, dJdbih, n_iteration):

        t = n_iteration

        self.momentum1_w = self.beta1*self.momentum1_w + (1-self.beta1) * dJdW
        self.momentum1_u = self.beta1*self.momentum1_u + (1-self.beta1) * dJdU
        self.momentum1_v = self.beta1*self.momentum1_v + (1-self.beta1) * dJdV
        self.momentum1_bhh = self.beta1*self.momentum1_bhh + (1-self.beta1) * dJdbhh
        self.momentum1_bho = self.beta1*self.momentum1_bho + (1-self.beta1) * dJdbho
        self.momentum1_bih = self.beta1*self.momentum1_bih + (1-self.beta1) * dJdbih




        self.momentum2_w = self.beta2*self.momentum2_w + (1-self.beta2) * (dJdW**2)
        self.momentum2_u = self.beta2*self.momentum2_u + (1-self.beta2) * (dJdU**2)
        self.momentum2_v = self.beta2*self.momentum2_v + (1-self.beta2) * (dJdV**2)
        self.momentum2_bhh = self.beta2*self.momentum2_bhh + (1-self.beta2) * (dJdbhh**2)
        self.momentum2_bih = self.beta2*self.momentum2_bih + (1-self.beta2) * (dJdbih**2)
        self.momentum2_bho = self.beta2*self.momentum2_bho + (1-self.beta2) * (dJdbho**2)


        mu1_w = self.momentum1_w/(1-self.beta1**t)
        mu1_u = self.momentum1_u/(1-self.beta1**t)
        mu1_v = self.momentum1_v/(1-self.beta1**t)
        mu1_bhh = self.momentum1_bhh/(1-self.beta1**t)
        mu1_bih = self.momentum1_bih/(1-self.beta1**t)
        mu1_bho = self.momentum1_bho/(1-self.beta1**t)


        mu2_w = self.momentum2_w/(1-self.beta2**t)
        mu2_u = self.momentum2_u/(1-self.beta2**t)
        mu2_v = self.momentum2_v/(1-self.beta2**t)
        mu2_bhh = self.momentum2_bhh/(1-self.beta2**t)
        mu2_bih = self.momentum2_bih/(1-self.beta2**t)
        mu2_bho = self.momentum2_bho/(1-self.beta2**t)

        self.W -= self.alpha * (mu1_w/np.sqrt(mu2_w+self.offset))
        self.U -= self.alpha * (mu1_u/np.sqrt(mu2_u+self.offset))
        self.V -= self.alpha * (mu1_v/np.sqrt(mu2_v+self.offset))
        self.bhh -= self.alpha * (mu1_bhh/np.sqrt(mu2_bhh+self.offset))
        self.bih -= self.alpha * (mu1_bih/np.sqrt(mu2_bih+self.offset))
        self.bho -= self.alpha * (mu1_bho/np.sqrt(mu2_bho+self.offset))



    def miniBatchGd(self,X,y):
        n_epochs = params['epochs']
        zipped = zip(X,y)
        num_cores = 0
        pool_size = 0
        J = -1
        obj = preprocess()
        word_to_index = obj.word_to_index
        parallel_flag = params["process_parallel"]
        if parallel_flag == "True":
            parallel_flag = True
        else:
            parallel_flag = False
        if(parallel_flag):
            num_cores = multiprocessing.cpu_count()
            pool_size = self.batch_size/num_cores
        for epochs in xrange(n_epochs):
            if(epochs%2==0):
                #forward propogate and get the loss
                prev_hidden = np.zeros(self.hidden_nodes)
                output, _ = self.forwardProp(X,prev_hidden)
                L = self.softmaxLoss(softmaxLoss, y)
                print "Epoch: "+str(epochs)+" over all Loss: "+str(self.L])+" time: "+str(time.time()-start)
                self.losses_after_epochs.append(L)

            if(epochs%5==0):
                print "Epoch: "+str(epochs)+" Loss: "+str(self.losses[-1])+" time: "+str(time.time()-start)
                sys.stdout.flush()
                print self.generateSent(word_to_index, 5)
                with open("controlTraining.txt",'r') as f:
                    control = f.read()
                    if control.strip() == "1":
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
                    self.tbpttTrainParallel(X_mini,y_mini,parallel_flag,num_cores,pool_size)
                else:
                    self.tbpttTrain(X_mini,y_mini,parallel_flag,num_cores,pool_size)


            #decay the learning rate
            self.alpha = 1.0*self.alpha/(1+epochs)

        print "Loss After Training "+str(self.losses[-1])
        sys.stdout.flush()




if __name__ == '__main__':
    model = RNNModel()
    model.train()
    pickle_file_sampled_data = open('pickledfiles/'+MODEL_FILE,'w')
    pickle.dump(model, pickle_file_sampled_data)
    pickle_file_sampled_data.close()
    print("--- Training completed in seconds %s---" % (time.time() - start))
