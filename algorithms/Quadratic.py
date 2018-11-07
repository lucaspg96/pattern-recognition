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
    
    def __init__(self,check_invertibility=False,pinv_mode="friedman"):
        self.check_invertibility = check_invertibility
        self.pinv_mode = pinv_mode
    
    def fit(self,X,Y,itter=100):
        classes = {}
        for (x,y) in zip(X,Y):
            if not y in classes:
                classes[y] = []
            
            classes[y].append(x)
        
        self.cells = {}
        
        if self.check_invertibility:
            need_pinv = False
            for k in classes:
                data = np.array(classes[k])
                m = np.mean(data,axis=0)
                cov = ut.covariance(data.transpose())

                invertibility, message = ut.is_invertible(cov)
                if invertibility:
                    self.cells[k] = DistanceCell(m,np.linalg.inv(cov))
                else:
                    need_pinv = True
                    print(message)
                    break
            
            if need_pinv:
                if self.pinv_mode == "friedman":
                    print("Computing regularized covariances matrices")
                    covs = ut.friedman_regularization(.5,1,classes)
                    for k in classes:
                        m = np.mean(data,axis=0)
                        self.cells[k] = DistanceCell(m,np.linalg.inv(covs[k]))
                        
                elif self.pinv_mode == "pooled":
                    print("Computing pooled covariance matrix")
                    cov = ut.pooled_covariance(classes)
                    inv_cov = np.linalg.inv(cov)
                    for k in classes:
                        m = np.mean(data,axis=0)
                        self.cells[k] = DistanceCell(m,inv_cov)
                else:
                    raise Exception("Invalid pinv method: {}".format(self.pinv_mode))
            
        else:
            for k in classes:
                data = np.array(classes[k])
                m = np.mean(data,axis=0)
                cov = ut.covariance(data.transpose())
                self.cells[k] = DistanceCell(m,np.linalg.inv(cov))
                    
            
    def predict(self, x):
        distances = [(c,self.cells[c].calculate(x)) for c in self.cells]
        distances = sorted(distances,key=lambda x: x[1])
        return distances[0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)