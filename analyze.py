
import matplotlib.pyplot as plt
import numpy as np
from RNN import RNNModel
import pickle
from preprocessor import preprocess
from Gru import Gru

FILE_NAME = 'modelGruv1'
title_len = 16

picklefile = open('pickledfiles/models/'+FILE_NAME,'r')
obj = pickle.loads(picklefile.read())

def plotLoss():
    y = obj.losses_after_epochs[1:-1]
    print len(y)
    # print y
    x = [i for i in range(len(y))]

    plt.plot(x,y)
    plt.xlabel("Every 3rd iteration")
    plt.ylabel("Loss")
    plt.show()

def genSent():
    objPre = preprocess()
    objPre = objPre.load()
    sentences =  obj.generateSent(objPre.word_to_index,1000,objPre.index_to_word)
    print sentences[:5]
    print "writing "+str(len(sentences))+" news"
    write_line = '\n'.join(sentences)
    open(FILE_NAME+'_sentences','w').write(write_line.encode('utf-8'))

# genSent()
# plotLoss()
genSent()
