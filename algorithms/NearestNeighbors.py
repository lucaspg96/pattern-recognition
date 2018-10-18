from algorithms.LearningAlgorithm import LearningAlgorithm
from algorithms import utils as ut
import numpy as np
from collections import Counter


class KNNClassifier(LearningAlgorithm):
    
    def __init__(self,k=10, distanceAlgorithm=ut.euclidian_distance):
        self.k = k
        self.distance = distanceAlgorithm
    
    def fit(self,X,Y,itter=100):
        self.data = [(x,y) for x,y in zip(X,Y)]
        
    def predict(self,x):
        distances = [(d[1],self.distance(x,list(d[0]))) for d in self.data]
        distances = sorted(distances,key=lambda x: x[1])
        top_k = [c for c,_ in distances[:self.k]]
        counter = Counter(top_k)
        return counter.most_common(1)[0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)
    

class ClusterNNClassifier(LearningAlgorithm):
    
    def __init__(self, distanceAlgorithm=ut.euclidian_distance):
        self.distance = distanceAlgorithm
    
    def fit(self,X,Y,itter=100):   
        classes = {}
        for (x,y) in zip(X,Y):
            if not y in classes:
                classes[y] = []
            
            classes[y].append(x)
        
        self.clusters = {}
        for k in classes:
            data = np.array(classes[k])
            cluster = np.mean(data,axis=0)
            self.clusters[k] = cluster
            
    def predict(self,x):
        distances = [(c,self.distance(x,self.clusters[c])) for c in self.clusters]
        distances = sorted(distances,key=lambda x: x[1])
        return distances[0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)
    
class NNClassifier(KNNClassifier):
    def __init__(self, distanceAlgorithm=ut.euclidian_distance):
        KNNClassifier.__init__(self, k=1, distanceAlgorithm=distanceAlgorithm)