
import json
import numpy as np
import pickle


params = json.loads(open("params.json").read())

class preprocess():

    def __init__(self):
        self.train_sentences = []
        self.test_sentences = []
        self.validate_sentences = []



    def cleanAndSave(self):
        try:
            picklefile = open('pickledfiles/data','r')
            obj = pickle.loads(picklefile.read())
            return obj
        except Exception as e:
            print "didn't find the pickle file pickling now....."

        with open('timesofindia.json', 'r') as f:
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

        return None




if __name__ == '__main__':
    obj = preprocess()
    actualobj = obj.cleanAndSave()
    if actualobj == None:
        pickle_file_sampled_data = open('pickledfiles/data','w')
        pickle.dump(obj,pickle_file_sampled_data)
