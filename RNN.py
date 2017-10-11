import numpy as np
import pickle
from preprocessor import preprocess
import json
from joblib import Parallel, delayed
import time
import multiprocessing


p_file = open('params.json','r')
params = json.loads(p_file.read())
np.random.seed(10)
start = time.time()


MODEL_FILE = 'modelv1'
train_size = params['training_size']

def execPlaceHolder(self,X,y):
    self.tbpttTrain(X,y)

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
        self.truncate = params['truncation']

        #some values for adam optimization
        self.momentum1_w = np.zeros(self.w_shape)
        self.momentum1_u = np.zeros(self.u_shape)
        self.momentum1_v = np.zeros(self.v_shape)
        self.momentum1_bh = 0.0
        self.momentum1_bi = 0.0

        self.momentum2_w = np.zeros(self.w_shape)
        self.momentum2_u = np.zeros(self.u_shape)
        self.momentum2_v = np.zeros(self.v_shape)
        self.momentum2_bh = 0.0
        self.momentum2_bi = 0.0

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
        X = np.array(list(data.X_train[:train_size]))
        y = np.array(list(data.y_train[:train_size]))
        self.randomizeParams()
        print "Everything loaded starting training"
        self.miniBatchGd(X,y)

    #takes a The whole dataset as the input and forward propogates for that
    def forwardProp(self,X,prev_hidden):
        #time steps
        # this function receives a sentence (in terms of index 2d array)
        T = X.shape[-1]
        dp = X.shape[0]
        hidden_state_info_activated = np.zeros((dp,T+1,self.hidden_nodes)) #for every training example i have a hidden state info
        hidden_state_info = np.zeros((dp,T+1,self.hidden_nodes))
        hidden_state_info[:][-1] = prev_hidden
        output = np.zeros((dp,T,self.word_dim)) #fist one should not be considered, therefore indexing is from 1 to T
        for t in np.arange(0,T):
            curr_hidden_state_info = hidden_state_info[:,t-1,:].reshape(dp,1,self.hidden_nodes)
            hidden_to_hidden = self.bh + np.dot(curr_hidden_state_info,self.W.T).squeeze()
            input_to_hidden = self.bi + (self.U[:,X[:,t]]).T
            non_activated = hidden_to_hidden+input_to_hidden
            activated = self.tanh(non_activated)
            hidden_state_info[:,t] = non_activated
            hidden_state_info_activated[:,t] = activated
            curr_output = self.bh + np.dot(hidden_state_info_activated[:,t].reshape(dp,1,self.hidden_nodes),self.V.T).squeeze()
            probab = self.softmax(curr_output)
            output[:,t] = probab

        #returns the information for the whole dataset

        return output, hidden_state_info, hidden_state_info_activated


    # the predict function returns the calculated output by just calculating the max probability, x is the single sentence
    def predict(self,X):
        output, _ , _ = self.forwardProp(X)
        return np.argmax(output, axis=-1)

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
        tmp = np.random.randn(np.product(self.w_shape)+1) * np.sqrt(1.0/self.hidden_nodes)
        self.bh = tmp[0] #hidden layer bias
        self.W = tmp[1:].reshape(self.w_shape)
        tmp = np.random.randn(np.product(self.u_shape)+1) * np.sqrt(1.0/self.word_dim)
        self.bi = tmp[0] #input layer bias
        self.U = tmp[1:].reshape(self.u_shape)
        self.V = np.random.randn(self.v_shape[0],self.v_shape[1]) * np.sqrt(1.0/self.hidden_nodes)




    #this is tensorflow style tbtt (apply this with minibatch)

    #takes one data point as the input and returns gradient change with respect to the weights
    def tbpttTrain(self,X,y,flag, num_cores, pool_size):
        #X here will be a mini batch this can be parallelized in the main function
        prev_hidden = np.zeros(self.hidden_nodes)
        for t in range(0,X.shape[-1],self.truncate):
            y_predicted, hidden_state_info,hidden_state_info_activated = self.forwardProp(X[:,t:t+self.truncate],prev_hidden)
            prev_hidden = hidden_state_info[:][-2]
            J = self.softmaxLoss(y_predicted, y[:,t:t+self.truncate])
            self.losses.append(J)
            if self.update_count % 30 == 0:
                print "After Updates: "+str(self.update_count)+" updates inside bptt Loss: " +str(J)
            #now backpropogate
            dJdV, dJdW, dJdU, dJdbh, dJdbi  = self.tbptt(X[:,t:t+self.truncate], y[:,t:t+self.truncate], y_predicted,hidden_state_info,hidden_state_info_activated,(t+self.truncate-1)%self.truncate)
            self.update_count += 1
            self.updateParamsAdam(dJdV, dJdW, dJdU, dJdbh, dJdbi,self.update_count)





    def tbptt(self, X, y, y_predicted, hidden_state_info, hidden_state_info_activated,T):

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
        dJdbh = np.sum(dy)
        dJdbi = 0

        #in reverse direction
        for t in range(T, T-self.truncate-1,-1):
            dh[:,t] += np.dot(dy[:,t],self.V) * (1-hidden_state_info_activated[:,t]**2) #due to the ouput layer
            total =  np.sum(dh[:,t])
            dJdbh += total
            dJdbi += total
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


        return  dJdV, dJdW, dJdU, dJdbh, dJdbi



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


    def updateParamsAdam(self,dJdV, dJdW, dJdU, dJdbh, dJdbi,n_iteration):

        t = n_iteration

        self.momentum1_w = self.beta1*self.momentum1_w + (1-self.beta1) * dJdW
        self.momentum1_u = self.beta1*self.momentum1_u + (1-self.beta1) * dJdU
        self.momentum1_v = self.beta1*self.momentum1_v + (1-self.beta1) * dJdV
        self.momentum1_bh = self.beta1*self.momentum1_bh + (1-self.beta1) * dJdbh
        self.momentum1_bi = self.beta1*self.momentum1_bi + (1-self.beta1) * dJdbi



        self.momentum2_w = self.beta2*self.momentum2_w + (1-self.beta2) * (dJdW**2)
        self.momentum2_u = self.beta2*self.momentum2_u + (1-self.beta2) * (dJdU**2)
        self.momentum2_v = self.beta2*self.momentum2_v + (1-self.beta2) * (dJdV**2)
        self.momentum2_bh = self.beta2*self.momentum2_bh + (1-self.beta2) * (dJdbh**2)
        self.momentum2_bi = self.beta2*self.momentum2_bi + (1-self.beta2) * (dJdbi**2)


        mu1_w = self.momentum1_w/(1-self.beta1**t)
        mu1_u = self.momentum1_u/(1-self.beta1**t)
        mu1_v = self.momentum1_v/(1-self.beta1**t)
        mu1_bh = self.momentum1_bh/(1-self.beta1**t)
        mu1_bi = self.momentum1_bi/(1-self.beta1**t)


        mu2_w = self.momentum2_w/(1-self.beta2**t)
        mu2_u = self.momentum2_u/(1-self.beta2**t)
        mu2_v = self.momentum2_v/(1-self.beta2**t)
        mu2_bh = self.momentum2_bh/(1-self.beta2**t)
        mu2_bi = self.momentum2_bi/(1-self.beta2**t)

        self.W -= self.alpha * (mu1_w/np.sqrt(mu2_w+self.offset))
        self.U -= self.alpha * (mu1_u/np.sqrt(mu2_u+self.offset))
        self.V -= self.alpha * (mu1_v/np.sqrt(mu2_v+self.offset))
        self.bh -= self.alpha * (mu1_bh/np.sqrt(mu2_bh+self.offset))
        self.bi -= self.alpha * (mu1_bi/np.sqrt(mu2_bi+self.offset))



    def miniBatchGd(self,X,y):
        n_epochs = params['epochs']
        zipped = zip(X,y)
        num_cores = 0
        pool_size = 0
        J = -1
        parallel_flag = params["process_parallel"]
        if parallel_flag == "True":
            parallel_flag = True
        else:
            parallel_flag = False
        if(parallel_flag):
            num_cores = multiprocessing.cpu_count()
            pool_size = self.batch_size/num_cores
        for epochs in xrange(n_epochs):
            if(epochs%5==0):
                print "Epoch: "+str(epochs)+" Loss: "+str(self.losses[-1])

            np.random.shuffle(zipped)
            X,y = zip(*zipped)
            X = np.array(X)
            y = np.array(y)
            for i in xrange(0,X.shape[0],self.batch_size):
                #get the current mini batch
                X_mini = X[i:i+self.batch_size]
                y_mini = y[i:i+self.batch_size]
                self.tbpttTrain(X_mini,y_mini,parallel_flag,num_cores,pool_size)


        print "Loss After Training "+str(self.losses[-1])



if __name__ == '__main__':
    model = RNNModel()
    model.train()
    pickle_file_sampled_data = open('pickledfiles/'+MODEL_FILE,'w')
    pickle.dump(model, pickle_file_sampled_data)
    pickle_file_sampled_data.close()
    print("--- Training completed in seconds %s---" % (time.time() - start))
