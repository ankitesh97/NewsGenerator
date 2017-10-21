
import json
import numpy as np
import pickle
import re
import nltk
import itertools
import time


params = json.loads(open("params.json").read())
link_to_replace_with = " https//examplearticle/exres/abcd.com "
twitter_link_to_replace = " imgtwittercom/abcdxyz "
VOCAB_SIZE = params['preprocess']['vocab_size']
SENTENCE_START = 'SENTENCE_START'
SENTENCE_END = 'SENTENCE_END'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
FILE_NAME = 'dataVectorizedv1.1'

class preprocess():

    def __init__(self):

        self.data = []
        self.X = []
        self.y = []

        # this contains the actual data in numerical
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.X_validate = []
        self.y_validate = []

        # meta data about the vocab
        self.index_to_word = []
        self.word_to_index = {}
        self.unknown = []
        self.vocab = []


    def makeData(self):
        self.getData()
        self.replaceAllLinks()
        self.replaceTwitterLinks()
        self.makeXy()
        self.splitData()
        self.equalize()


    def load(self):
        picklefile = open('pickledfiles/'+FILE_NAME,'r')
        obj = pickle.loads(picklefile.read())
        return obj

    def getData(self):

        with open('pickledfiles/timesofindia.json', 'r') as f:
            data = json.loads(f.read())[:params["preprocess"]["total"]]
            l = len(data)
            total = []
            for entry in data:
                total.append(entry["text"])

            np.random.shuffle(total)
            self.data = total



    def replaceAllLinks(self):
        pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        compiled = re.compile(pattern)
        cleaned_data = []
        for sent in self.data:
            cleaned_data.append(compiled.sub(link_to_replace_with,sent))

        self.data = cleaned_data

    def replaceTwitterLinks(self):

        pattern = "[a-zA-z]+.twitter.com/[a-zA-Z0-9]+"
        compiled = re.compile(pattern)
        cleaned_data = []
        for sent in self.data:
            cleaned_data.append(compiled.sub(twitter_link_to_replace,sent))

        self.data = cleaned_data

    def makeXy(self):
        train = []
        test = []
        validate = []
        data = []
        #add start token and end token
        for sent in self.data:
            data.append(SENTENCE_START+" "+sent+" "+SENTENCE_END)

        #this contains tokens for all the sentences
        tokenized = [nltk.word_tokenize(data[i]) for i in range(len(data))]
        frequency = nltk.FreqDist(itertools.chain(*tokenized))
        self.vocab = frequency.most_common(VOCAB_SIZE-1)
        #CREATE TWO MAPPING WORD_TO_INDEX AND INDEX_TO_WORD
        #we also have an UNKNOWN_TOKEN i.e if the word is not found in our vocab then we replace it by UNKNOWN
        self.index_to_word = [x[0] for x in self.vocab]
        self.index_to_word.append(UNKNOWN_TOKEN)
        for i,w in enumerate(self.index_to_word):
            self.word_to_index[w] = i
        #
        # #replace the words that are not in our vocab with the UNKNOWN_TOKEN
        #
        for i in xrange(len(tokenized)):
            for j in xrange(len(tokenized[i])):
                if(self.word_to_index.has_key(tokenized[i][j])==False):
                    self.unknown.append(tokenized[i][j])
                    tokenized[i][j] = UNKNOWN_TOKEN
        #

        #
        #now to create arrays of the input
        for i in xrange(len(tokenized)):
            X,y = self.convertToXy(tokenized[i])
            self.X.append(X)
            self.y.append(y)




    def convertToXy(self, tokenized):
        X = []
        y = []
        for i in xrange(len(tokenized)-1):
            X.append(self.word_to_index[tokenized[i]])
            y.append(self.word_to_index[tokenized[i+1]])

        return np.array(X),np.array(y)

    def splitData(self):
        total = len(self.X)
        n_train = int(float(params['preprocess']['train']) * total)
        n_test = int(float(params['preprocess']['train']) * total)

        self.X_train = self.X[:n_train]
        self.X_test = self.X[n_train:n_train+n_test]
        self.X_validate = self.X[n_train+n_test:]

        self.y_train = self.y[:n_train]
        self.y_test = self.y[n_train:n_train+n_test]
        self.y_validate = self.y[n_train+n_test:]

    def equalize(self):
        # print self.word_to_index
        max_l = 0
        for x in self.X_train:
            max_l = max(len(x),max_l)
        end = self.word_to_index["SENTENCE_END"]
        for i in range(len(self.X_train)):
            l = max_l - len(self.X_train[i])
            to_append = [end]*l
            self.X_train[i] = np.concatenate((self.X_train[i],to_append))
            self.y_train[i] = np.concatenate((self.y_train[i],to_append))

        max_l = 0
        for x in self.X_test:
            max_l = max(len(x),max_l)

        for i in range(len(self.X_test)):
            l = max_l - len(self.X_test[i])
            to_append = [end]*l
            self.X_test[i] = np.concatenate((self.X_test[i],to_append))
            self.y_test[i] = np.concatenate((self.y_test[i],to_append))

        max_l = 0
        for x in self.X_validate:
            max_l = max(len(x),max_l)

        for i in range(len(self.X_validate)):
            l = max_l - len(self.X_validate[i])
            to_append = [end]*l
            self.X_validate[i] = np.concatenate((self.X_validate[i],to_append))
            self.y_validate[i] = np.concatenate((self.y_validate[i],to_append))



if __name__ == '__main__':
    start = time.time()
    obj = preprocess()
    obj.makeData()
    pickle_file_sampled_data = open('pickledfiles/'+FILE_NAME,'w')
    pickle.dump(obj,pickle_file_sampled_data)
    print "seconds ---------- "+str(time.time()-start)
