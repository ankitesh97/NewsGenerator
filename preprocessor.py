
import json
import numpy as np
import pickle
import re
import nltk
import itertools
import time


params = json.loads(open("params.json").read())
link_to_replace_with = " https://examplearticle/exres/abcd.com "
twitter_link_to_replace = " img.twitter.com/abcdxyz "
VOCAB_SIZE = params['preprocess']['vocab_size']
SENTENCE_START = 'SENTENCE_START'
SENTENCE_END = 'SENTENCE_END'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
FILE_NAME = 'dataVectorized'

class preprocess():

    def __init__(self):
        # this will contain the raw senteneces
        self.train_sentences = []
        self.test_sentences = []
        self.validate_sentences = []

        # this will contained the cleaned sentences

        self.train = []
        self.test =[]
        self.validate = []

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
        self.clean()
        self.replaceAllLinks()
        self.replaceTwitterLinks()
        self.makeXy()


    def load(self):
        picklefile = open('pickledfiles/'+FILE_NAME,'r')
        obj = pickle.loads(picklefile.read())
        return obj

    def clean(self):

        with open('pickledfiles/timesofindia.json', 'r') as f:
            data = json.loads(f.read())
            l = len(data)
            total = []
            for entry in data:
                total.append(entry["text"])

            np.random.shuffle(total)
            p = params["preprocess"]
            train = int(p["train"]*l)
            test = int(p["test"]*l)
            self.train_sentences = total[:train]
            self.test_sentences = total[train:test]
            self.validate_sentences = total[test:]



    def replaceAllLinks(self):
        pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        compiled = re.compile(pattern)
        for sent in self.train_sentences:
            self.train.append(compiled.sub(link_to_replace_with,sent))

        for sent in self.test_sentences:
            self.test.append(compiled.sub(link_to_replace_with,sent))

        for sent in self.validate_sentences:
            self.validate.append(compiled.sub(link_to_replace_with,sent))

    def replaceTwitterLinks(self):
        train = []
        test = []
        validate = []
        pattern = "[a-zA-z]+.twitter.com/[a-zA-Z0-9]+"
        compiled = re.compile(pattern)
        for sent in self.train:
            train.append(compiled.sub(twitter_link_to_replace,sent))

        for sent in self.test:
            test.append(compiled.sub(twitter_link_to_replace,sent))

        for sent in self.validate:
            validate.append(compiled.sub(twitter_link_to_replace,sent))

        self.train = train
        self.test = test
        self.validate = validate


    def makeXy(self):
        train = []
        test = []
        validate = []

        #add start token and end token
        for sent in self.train:
            train.append(SENTENCE_START+" "+sent+" "+SENTENCE_END)

        for sent in self.test:
            test.append(SENTENCE_START+" "+sent+" "+SENTENCE_END)

        for sent in self.validate:
            validate.append(SENTENCE_START+" "+sent+" "+SENTENCE_END)

        self.train = train
        self.test = test
        self.validate = validate
        del train
        del test
        del validate
        #this contains tokens for all the sentences
        tokenized = [nltk.word_tokenize(self.train[i]) for i in range(len(self.train))]
        tokenized_test = [nltk.word_tokenize(self.test[i]) for i in range(len(self.test))]
        tokenized_validate = [nltk.word_tokenize(self.validate[i]) for i in range(len(self.validate))]
        frequency = nltk.FreqDist(itertools.chain(*tokenized))
        self.vocab = frequency.most_common(VOCAB_SIZE-1)
        #CREATE TWO MAPPING WORD_TO_INDEX AND INDEX_TO_WORD
        #we also have an UNKNOWN_TOKEN i.e if the word is not found in our vocab then we replace it by UNKNOWN
        self.index_to_word = [x[0] for x in self.vocab]
        self.index_to_word.append(UNKNOWN_TOKEN)
        for i,w in enumerate(self.index_to_word):
            self.word_to_index[w] = i

        #replace the words that are not in our vocab with the UNKNOWN_TOKEN

        for i in xrange(len(tokenized)):
            for j in xrange(len(tokenized[i])):
                if(self.word_to_index.has_key(tokenized[i][j])==False):
                    self.unknown.append(tokenized[i][j])
                    tokenized[i][j] = UNKNOWN_TOKEN

        for i in xrange(len(tokenized_test)):
            for j in xrange(len(tokenized_test[i])):
                if(self.word_to_index.has_key(tokenized_test[i][j])==False):
                    self.unknown.append(tokenized_test[i][j])
                    tokenized_test[i][j] = UNKNOWN_TOKEN


        for i in xrange(len(tokenized_validate)):
            for j in xrange(len(tokenized_validate[i])):
                if(self.word_to_index.has_key(tokenized_validate[i][j])==False):
                    self.unknown.append(tokenized_validate[i][j])
                    tokenized_validate[i][j] = UNKNOWN_TOKEN



        #now to create arrays of the input
        for i in xrange(len(tokenized)):
            X,y = self.convertToXy(tokenized[i])
            self.X_train.append(X)
            self.y_train.append(y)


        for i in xrange(len(tokenized_test)):
            X,y = self.convertToXy(tokenized_test[i])
            self.X_test.append(X)
            self.y_test.append(y)

        for i in xrange(len(tokenized_validate)):
            X,y = self.convertToXy(tokenized_validate[i])
            self.X_validate.append(X)
            self.y_validate.append(y)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        self.X_validate = np.array(self.X_validate)
        self.y_validate = np.array(self.y_validate)
        print self.X_train.shape, self.y_train.shape



    def convertToXy(self, tokenized):
        X = []
        y = []
        for i in xrange(len(tokenized)-1):
            X.append(self.word_to_index[tokenized[i]])
            y.append(self.word_to_index[tokenized[i+1]])

        return np.array(X),np.array(y)



def pickleJsonEqual():
    obj = preprocess().load()
    #
    max_l = 0
    for x in obj.X_train:
        max_l = max(len(x),max_l)
    end = obj.word_to_index["SENTENCE_END"]
    for i in range(len(obj.X_train)):
        l = max_l - len(obj.X_train[i])
        to_append = [end]*l
        obj.X_train[i] = np.concatenate((obj.X_train[i],to_append))
        obj.y_train[i] = np.concatenate((obj.y_train[i],to_append))

    max_l = 0
    for x in obj.X_test:
        max_l = max(len(x),max_l)

    for i in range(len(obj.X_test)):
        l = max_l - len(obj.X_test[i])
        to_append = [end]*l
        obj.X_test[i] = np.concatenate((obj.X_test[i],to_append))
        obj.y_test[i] = np.concatenate((obj.y_test[i],to_append))

    max_l = 0
    for x in obj.X_validate:
        max_l = max(len(x),max_l)

    for i in range(len(obj.X_validate)):
        l = max_l - len(obj.X_validate[i])
        to_append = [end]*l
        obj.X_validate[i] = np.concatenate((obj.X_validate[i],to_append))
        obj.y_validate[i] = np.concatenate((obj.y_validate[i],to_append))


    pickle_file_sampled_data = open('pickledfiles/dataVectorized','w')
    pickle.dump(obj,pickle_file_sampled_data)


if __name__ == '__main__':
    start = time.time()
    obj = preprocess()
    obj.makeData()
    pickle_file_sampled_data = open('pickledfiles/'+FILE_NAME,'w')
    pickle.dump(obj,pickle_file_sampled_data)
    print "seconds ---------- "+str(time.time()-start)
