"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
from Model import *
from Features import *
import pickle

class Perceptron(Model):
    def __init__(self,model_file):
        super().__init__(model_file)
          
    def predict(self,w,X,classes):
        pred=[]
        for i,x in enumerate(X):
            temp= np.dot(w,x)
            h=np.argmax(temp)
            pred.append(classes[np.argmax(temp)])
        
        return np.array(pred)
    
    def accuracy(self,y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        return accuracy
    
    def tokenize(self,text):
        # TODO customize to your needs
        text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        text=text.lower()
        return text.split()
    
    
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
        print(self.classes)
        self.weights = np.zeros((self.n_class,self.n_features),dtype=np.float64)
        
        print("..............Training.............")
        for j in range(20):
            for i,x in enumerate(self.X):
                pred=np.dot(self.weights,x.T)
                idx=np.argmax(pred)
                label=self.classes[idx]
                correct_idx = list(self.classes).index(self.y[i])
                if label!=self.y[i]:
                    self.weights[idx]=self.weights[idx]-x
                    self.weights[correct_idx]=self.weights[correct_idx]+x
                
            preds=self.predict(self.weights,self.X,self.classes)
            print("training accuracy:", self.accuracy(self.y, preds))
            
        ## TODO write your code here
        model = {}
        model["weights"]=self.weights
        model["idf"]=self.idf
        model["classes"]=self.classes
        ## Save the model
        self.save_model(model)
        return model

    def remove_oov(self,tokenized_text,vocab_list):
        for i,doc in enumerate(tokenized_text):
            temp=list(tokenized_text[i])
            for word in doc:
                if (word not in vocab_list):
                    temp.remove(word)
            tokenized_text[i]=temp
        return tokenized_text
    
    def calculate_tfIdf(self,tokenized_text, idf_scores,X):
        
        vocab_count_dict=dict.fromkeys(idf_scores.keys(),0)
        for i,document in enumerate(tokenized_text):
            for word in document:
                    vocab_count_dict[word]+=1
            sum_idf=0            
            for keys in vocab_count_dict:
                vocab_count_dict[keys]=vocab_count_dict[keys]*idf_scores[keys]
                sum_idf+= vocab_count_dict[keys]**2
             
            sum_idf=np.sqrt(sum_idf)
            
            for keys in vocab_count_dict:
                vocab_count_dict[keys]=vocab_count_dict[keys]/sum_idf
                
            X[i] = np.array(list(vocab_count_dict.values()))
            vocab_count_dict=dict.fromkeys(idf_scores.keys(),0)
         
        return X
        
    def bag_of_words(self, tokenized_text, vocab, X):
        vocab_count=dict.fromkeys(vocab,0)
        for i,document in enumerate(tokenized_text):
            for word in document:
                if word in vocab:
                    vocab_count[word]+=1
            X[i]=np.array(list(vocab_count.values()))
            vocab_count_dict=dict.fromkeys(vocab,0)
        
        return X
        
        
    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """
        ## TODO write your code here (and change return)
        
        
        idf_scores=model['idf']
        weights = model["weights"]
        classes = model["classes"]
        print(classes)
        with open(input_file,encoding="utf8") as file:
            data = file.read().splitlines()

        # data_split = map(methodcaller("rsplit", "\t", 1), data)
        # texts, labels = map(list, zip(*data_split))
        vocab_list= list(idf_scores.keys())
        tokenized_text = [self.tokenize(text) for text in data]
        tokenized_text=self.remove_oov(tokenized_text, vocab_list)
        
        n_features=len(vocab_list)
        n_size = len(data)
        X=np.zeros((n_size,n_features))
        vocab=set(idf_scores.keys())
        X=self.calculate_tfIdf(tokenized_text, idf_scores, X)
        #X=self.bag_of_words(tokenized_text, vocab, X)
        preds=self.predict(weights,X,classes)
        # print("validation accuracy:", self.accuracy(labels, preds))
        
        # print(preds)
        
        return preds
    
# p=Perceptron("C:/USC notes/CSCI 662/assignments/assignment 1/perceptron.products.model")
# model=p.train("datasets/products-train.txt")
#
# preds=p.classify("datasets/products-test.txt", model)
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
