"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y} 
Please refer to lecture notes Chapter 4 for more details
"""

from Model import *
from Features import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class NaiveBayes(Model):
    def __init__(self,model_file):
        super().__init__(model_file)
        self.model = None

    def tokenize(self,text):
        # TODO customize to your needs
        text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        text=text.lower()
        return text.split()

    def remove_oov(self,tokenized_text,vocab_list):
        for i,doc in enumerate(tokenized_text):
            temp=list(tokenized_text[i])
            for word in doc:
                if (word not in vocab_list):
                    temp.remove(word)
            tokenized_text[i]=temp
        return tokenized_text

    def calculate_tfIdf(self, tokenized_text, idf_scores, X):

        vocab_count_dict = dict.fromkeys(idf_scores.keys(), 0)
        for i, document in enumerate(tokenized_text):
            for word in document:
                vocab_count_dict[word] += 1
            sum_idf = 0
            for keys in vocab_count_dict:
                vocab_count_dict[keys] = vocab_count_dict[keys] * idf_scores[keys]
                sum_idf += vocab_count_dict[keys] ** 2

            sum_idf = np.sqrt(sum_idf)

            for keys in vocab_count_dict:
                vocab_count_dict[keys] = vocab_count_dict[keys] / sum_idf

            X[i] = np.array(list(vocab_count_dict.values()))
            vocab_count_dict = dict.fromkeys(idf_scores.keys(), 0)

        return X

    def bag_of_words(self, tokenized_text, vocab, X):
        vocab_count = dict.fromkeys(vocab, 0)
        for i, document in enumerate(tokenized_text):
            for word in document:
                if word in vocab:
                    vocab_count[word] += 1
            X[i] = np.array(list(vocab_count.values()))
            vocab_count_dict = dict.fromkeys(vocab, 0)

        return X

    def calculate_priors_mean_var(self,X,y, classes):
        n_class = len(classes)
        priors= np.zeros(n_class, dtype=np.float64)
        for i, label in enumerate(classes):
            self.mean[i,:]=np.mean(X[y==label], axis=0)
            self.var[i,:]=np.var(X[y==label], axis=0)
            self.priors[i]= len(y[y==label]) / len(y)
        
    
    def calculate_mean_var(self,X):
        mean = np.mean(X,axis=0)
        var = np.var(X,axis=0)
        return mean,var
    
    def likelihood(self,class_id,x, mean, var):
        mean= mean[class_id]
        var=var[class_id]
        
        numerator = np.exp(-((x - mean) ** 2) / (2 * (var) ))
        denominator = np.sqrt(2 * np.pi * (var))
        p=numerator/denominator
        return numerator/denominator
    
    def posterior_probability(self,x,priors,classes, mean,var):
        posterior_pr=[]
        
        for i,c in enumerate(classes):
            prior_pr= np.log(priors[i])
            posterior= np.nansum(np.log(self.likelihood(i, x, mean,var)))
            posterior=prior_pr+posterior
            posterior_pr.append(posterior)
            
        return classes[np.argmax(posterior_pr)]
            
    def predict(self,X,priors,classes, mean,var):
        y_pred = [self.posterior_probability(x,priors,classes, mean,var) for x in X]
        return np.array(y_pred)
    
    def accuracy(self,y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        f=Features(input_file)
        self.X, self.y, self.idf = f.get_features()
        self.X=np.array(self.X)
        self.y = np.array(self.y)
        self.n_data, self.n_features = self.X.shape
        print("..........features extracted.......")
        self.classes = np.unique(self.y)
        self.n_class = len(self.classes)
        
        self.mean = np.zeros((self.n_class, self.n_features), dtype=np.float64)
        self.var = np.zeros((self.n_class, self.n_features), dtype=np.float64)
        self.priors = np.zeros(self.n_class, dtype=np.float64)
        print("..............Training.............")
        self.calculate_priors_mean_var(self.X,self.y, self.classes)
        
        y_pred = self.predict(self.X,self.priors,self.classes, self.mean, self.var)
        
        print(self.accuracy(self.y,y_pred))
        
        ## TODO write your code here
        model = {}
        model["priors"] = self.priors
        model['mean']= self.mean
        model["var"]=self.var
        model["idf"] = self.idf
        model["classes"] = self.classes
        model["type"]="tfidf"

        ## Save the model
        self.save_model(model)
        return model


    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """


        ## TODO write your code here
        priors = model["priors"]
        mean = model['mean']
        var = model['var']
        idf_scores=model['idf']
        classes=model["classes"]

        print(classes)
        with open(input_file, encoding="utf8") as file:
            data = file.read().splitlines()

        # data_split = map(methodcaller("rsplit", "\t", 1), data)
        # texts, labels = map(list, zip(*data_split))
        vocab_list = list(idf_scores.keys())
        tokenized_text = [self.tokenize(text) for text in data]
        tokenized_text = self.remove_oov(tokenized_text, vocab_list)

        n_features = len(vocab_list)
        n_size = len(data)
        X = np.zeros((n_size, n_features))
        vocab = set(idf_scores.keys())
        if model['type']=="tfidf":
            X = self.calculate_tfIdf(tokenized_text, idf_scores, X)
        else:
            X = self.bag_of_words(tokenized_text, vocab, X)

        preds = self.predict(X,priors,classes, mean, var)
        return preds


# nb=NaiveBayes("C:/USC notes/CSCI 662/assignments/assignment 1/nb.products.model")
# model=nb.train("datasets/products-train.txt")

# preds=nb.classify("datasets/products-test.txt", model)
# with open("datasets/products-label-test.txt",encoding="utf8") as file:
#     data = file.readlines()
#
# label=[]
# for i in range(len(data)):
#     label.append(data[i].strip())
#
#
# label=np.array(label)
#
# acc = np.mean(preds==label)
# print(acc)