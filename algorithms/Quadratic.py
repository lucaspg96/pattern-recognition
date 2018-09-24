from algorithms.LearningAlgorithm import LearningAlgorithm
from algorithms import utils as ut
import numpy as np

class DistanceCell():
        def __init__(self,m,Q):
            self.m = m
            self.distance = ut.generate_mahalanobis_distance(Q)
            
        def calculate(self, x):
            return self.distance(self.m,x)

class QuadraticClassifier(LearningAlgorithm):
    
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
            covariance = ut.covariance(data.transpose())
            self.cells[k] = DistanceCell(m,np.linalg.inv(covariance))
            
    def predict(self, x):
        distances = [(c,self.cells[c].calculate(x)) for c in self.cells]
        distances = sorted(distances,key=lambda x: x[1])
        return distances[0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)