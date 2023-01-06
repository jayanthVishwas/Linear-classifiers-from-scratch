"""
    Basic feature extractor
"""
from operator import methodcaller
import string
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    text=text.lower()
    return text.split()

class Features:

    def __init__(self, data_file):
        with open(data_file,encoding="utf8") as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, self.labels = map(list, zip(*data_split))
        self.vocab=set()
        self.tokenized_text = [tokenize(text) for text in texts]
        self.labelset = list(set(self.labels))
        self.word_freq={}
        self.bof_check=False

        
        if "products" in data_file:
            print("bow")
            self.bof_check=True
            self.word_freq = self.get_word_frequecny(self.tokenized_text)
            self.word_freq = sorted(self.word_freq.items(), key=lambda x:x[1], reverse=True)
            self.word_freq = dict(list(self.word_freq)[0:12000])
            self.vocab= set(self.word_freq.keys())
            self.vocab_count_dict=dict.fromkeys(self.vocab,0)
            
            self.features=len(self.word_freq)
            self.size = len(self.labels)
            self.X = np.zeros((self.size,self.features))
            self.X = self.bag_of_words(self.tokenized_text, self.vocab_count_dict,self.vocab, self.X)
            
        else:
            
            self.vocab= self.create_vocab()
            self.vocab_count_dict=dict.fromkeys(self.vocab,0)
            self.dict_idf=dict.fromkeys(self.vocab,0)
            self.dict_idf=self.calculate_IDF()
            
            self.features=len(self.vocab)
            self.size = len(self.labels)
            self.X=np.zeros((self.size,self.features))
    
            self.calculate_tfIdf()
            
               

    #@classmethod
    def get_word_frequecny(self,tokenized_text):
        word_freq={}
        for token in tokenized_text:
            for word in token:
                if word in word_freq:
                    word_freq[word]+=1
                else:
                    word_freq[word]=1
        return word_freq
        
    def bag_of_words(self,tokenized_text, vocab_count,vocab,X):
        
        for i,document in enumerate(tokenized_text):
            for word in document:
                if word in vocab:
                    vocab_count[word]+=1
            X[i]=np.array(list(vocab_count.values()))
            vocab_count_dict=dict.fromkeys(self.vocab,0)
        
        return X
        
        
        
        
    def get_features(self):
        # TODO: implement this method by implementing different classes for different features
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features
        sns.distplot(self.X, x=self.X[1])
        sns.distplot(self.X,x=self.X[5])
        plt.show()
        if self.bof_check:
            print("bow")
            return self.X, self.labels, self.word_freq
        else:
            return self.X, self.labels, self.dict_idf

    def create_vocab(self):
        for tokens in self.tokenized_text:
            for word in tokens:
                word=word.strip()
                self.vocab.add(word)
        return self.vocab

    def calculate_IDF(self):
        for key in self.dict_idf:
            for tokens in self.tokenized_text:
                if key in tokens:
                    self.dict_idf[key]+=1
                    continue
            self.dict_idf[key]= np.log((len(self.tokenized_text)+1) /(self.dict_idf[key]+1))
        return self.dict_idf

    def calculate_tfIdf(self):

        for i,document in enumerate(self.tokenized_text):
            for word in document:
                    self.vocab_count_dict[word]+=1
            sum_idf=0            
            for keys in self.vocab_count_dict:
                self.vocab_count_dict[keys]=self.vocab_count_dict[keys]*self.dict_idf[keys]
                sum_idf+= self.vocab_count_dict[keys]**2
             
            sum_idf=np.sqrt(sum_idf)
            
            for keys in self.vocab_count_dict:
                self.vocab_count_dict[keys]=self.vocab_count_dict[keys]/sum_idf
                
            self.X[i] = np.array(list(self.vocab_count_dict.values()))
            self.vocab_count_dict=dict.fromkeys(self.vocab,0)


# f = Features("datasets/4dim-train.txt")
# x,y,z=f.get_features()
# print("features extracted")
# print(x)
# print(x.shape)