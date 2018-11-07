from algorithms.LearningAlgorithm import LearningAlgorithm
import numpy as np
import math
from algorithms import utils as ut

class LogisticRegressionClassifier(LearningAlgorithm):
    def __init__(self,alpha=.25):
        self.alpha = alpha
    
    def __logistic_function(self,x):
        return 1/(np.ones(self.weights.shape[0])+np.exp(-1*self.weights.dot(x.T)))
    
    def fit(self,X,Y,itter=100):
        self.classes_to_num = {}
        classes = []
        
        for y in Y:
            if not y in self.classes_to_num:
                self.classes_to_num[y] = len(self.classes_to_num)
            
            classes.append(self.classes_to_num[y])
            
        self.num_to_classes = {self.classes_to_num[k]:k for k in self.classes_to_num}
        classes = ut.one_hot_encode(classes)
        
        X = np.c_[np.ones(len(X)),X].astype(float)
        self.weights = np.random.randn(classes.shape[1],X.shape[1])

        n = len(X)
        for _ in range(100):
            for x,y in zip(X,classes):
                p = self.__logistic_function(x)
                self.weights += (self.alpha/n)*np.matmul((y-p).reshape(len(y),1),x.reshape(1,len(x)))
            
    
    def predict(self,X,label = True):
        X = np.concatenate([[1],X])
        y = np.argmax(self.__logistic_function(X))
        return self.num_to_classes[y]
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)