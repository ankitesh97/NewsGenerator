
import json
import numpy as np
import pickle
import re
import nltk
import itertools


params = json.loads(open("params.json").read())
link_to_replace_with = " https://examplearticle/exres/abcd.com "
twitter_link_to_replace = " img.twitter.com/abcdxyz "
VOCAB_SIZE = params['preprocess']['vocab_size']
SENTENCE_START = 'SENTENCE_START'
SENTENCE_END = 'SENTENCE_END'
UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'

class preprocess():

    def __init__(self):
        self.train_sentences = []
        self.test_sentences = []
        self.validate_sentences = []
        self.train = []
        self.test = []
        self.validate = []
        self.index_to_word = []
        self.word_to_index = {}
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.X_validate = []
        self.y_validate = []
        self.unknown = []
        self.vocab = []


    def callMeFirst(self,loaded_obj):
       self.train_sentences = loaded_obj.train_sentences
       self.test_sentences = loaded_obj.test_sentences
       self.validate_sentences = loaded_obj.validate_sentences
       self.train = loaded_obj.train
       self.test = loaded_obj.test
       self.validate = loaded_obj.validate


    def load(self):
        picklefile = open('pickledfiles/sentences2','r')
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
            train.append(SENTENCE_START+sent+SENTENCE_END)

        for sent in self.test:
            test.append(SENTENCE_START+sent+SENTENCE_END)

        for sent in self.validate:
            validate.append(SENTENCE_START+sent+SENTENCE_END)

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



if __name__ == '__main__':
    obj = preprocess()
    o = obj.load()
    newobj = preprocess()
    newobj.callMeFirst(o)
    newobj.makeXy()
    # # obj.clean()
    # # obj.replaceAllLinks()
    pickle_file_sampled_data = open('pickledfiles/dataRefined','w')
    pickle.dump(newobj,pickle_file_sampled_data)
