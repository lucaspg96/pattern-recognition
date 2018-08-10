from algorithms.LearningAlgorithm import LearningAlgorithm
from algorithms import utils as ut
import numpy as np
from collections import Counter


class SimpleKNNClassifier(LearningAlgorithm):
    
    def __init__(self,k=10, distanceAlgorithm=ut.euclidian_distance):
        self.k = k
        self.distance = distanceAlgorithm
    
    def fit(self,X,Y,itter=100):
        self.data = [(x,y) for x,y in zip(X,Y)]
        
    def predict(self,x):
        distances = [(d[1],self.distance(x,list(d[0]))) for d in self.data]
        distances = sorted(distances,key=lambda x: x[1])
        top_k = distances[:self.k]
        counter = Counter(top_k)
        return counter.most_common(1)[0][0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)