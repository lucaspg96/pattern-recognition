import numpy as np
from algorithms.utils import covariance
from algorithms.LearningAlgorithm import LearningAlgorithm
from math import sqrt, pi, exp, pow

class NormalNaiveBayes(LearningAlgorithm):
   
    def fit(self,X,Y,itter=100):   
        classes = {}
        for (x,y) in zip(X,Y):
            if not y in classes:
                classes[y] = []
            
            classes[y].append(x)
        
        self.cells = {}
        for k in classes:
            data = np.array(classes[k])
            m = np.mean(data,axis=0)
            std = np.std(data,axis=0)
            var = np.var(data,axis=0)
            self.cells[k] = {
                "icov": np.linalg.inv(np.diag(var)),
                "rtcov": sqrt(np.prod(var)),
                "mean": m,
                "std": std
            }
            
    def __prob__(self,x,k):
        s = self.cells[k]["std"]
        m = self.cells[k]["mean"]
        ic = self.cells[k]["icov"]
        rtc = self.cells[k]["rtcov"]
        z = x-m
        return (exp(np.matmul(z, np.matmul(ic,z)))*(-0.5))/(pow(2*pi,x.shape[0]/2)*rtc)
            
    def predict(self,x):
        distances = [(k,self.__prob__(x,k)) for k in self.cells]
        distances = sorted(distances,key=lambda x: x[1], reverse=True)
        return distances[0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)