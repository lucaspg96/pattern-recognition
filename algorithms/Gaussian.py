import numpy as np
from algorithms.utils import covariance
from algorithms.LearningAlgorithm import LearningAlgorithm
import math
from algorithms import utils as ut

class NormalNaiveBayes():
   
    def fit(self,X,Y):   
        classes = {}
        n = X.shape[0]
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
            i_var = np.array([1/v if not v == 0 else 0 for v in var])
            self.cells[k] = {
                "icov": i_var,
                "cov_det": np.prod(var),
                "mean": m,
                "std": std,
                "prob_priori": data.shape[0]/n
            }
            
    def __prob__(self,x,k):
        s = self.cells[k]["std"]
        m = self.cells[k]["mean"]
        ic = self.cells[k]["icov"]
        cov_det = self.cells[k]["cov_det"]
        prob_priori = self.cells[k]["prob_priori"]
        z = x-m
        if cov_det == 0:
            cov_det = 1
        return math.log(prob_priori) \
                - 0.5*z.dot(ic*z) - 0.5*math.log(cov_det)
            
    def predict(self,x):
        distances = [(k,self.__prob__(x,k)) for k in self.cells]
        distances = sorted(distances,key=lambda x: x[1], reverse=True)
        return distances[0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)
    
class QuadraticGaussianClassifier(LearningAlgorithm):

    def __init__(self,check_invertibility=False,pinv_mode="friedman"):
        self.check_invertibility = check_invertibility
        self.pinv_mode = pinv_mode

    def fit(self,X,Y,itter=100):   
        classes = {}
        n = X.shape[0]
        for (x,y) in zip(X,Y):
            if not y in classes:
                classes[y] = []
            
            classes[y].append(x)
        
        self.cells = {}
        
        if self.check_invertibility:
            for k in classes:
                data = np.array(classes[k])
                m = np.mean(data,axis=0)
                std = np.std(data,axis=0)
                cov = covariance(data.transpose())
                
                invertibility, message = ut.is_invertible(cov)
                if invertibility:
                    self.cells[k] = {
                        "icov": np.linalg.inv(cov),
                        "rtcov": math.sqrt(np.linalg.det(cov)),
                        "mean": m,
                        "std": std,
                        "prob_priori": data.shape[0]/n
                    }
                    
                else:
                    need_pinv = True
                    break
                
            if need_pinv:
                if self.pinv_mode == "friedman":
                    covs = ut.friedman_regularization(.5,1,classes)
                    for k in classes:
                        m = np.mean(data,axis=0)
                        std = np.std(data,axis=0)
                        cov = covs[k]
                        self.cells[k] = {
                            "icov": np.linalg.inv(cov),
                            "cov_det": np.linalg.det(cov),
                            "mean": m,
                            "std": std,
                            "prob_priori": data.shape[0]/n
                        }

                elif self.pinv_mode == "pooled":
                    cov = ut.pooled_covariance(classes)
                    inv_cov = np.linalg.inv(cov)
                    for k in classes:
                        m = np.mean(data,axis=0)
                        std = np.std(data,axis=0)
                        self.cells[k] = {
                            "icov":inv_cov,
                            "cov_det": np.linalg.det(cov),
                            "mean": m,
                            "std": std,
                            "prob_priori": data.shape[0]/n
                        }
                else:
                    raise Exception("Invalid pinv method: {}".format(self.pinv_mode))
            
        else:
            for k in classes:
                data = np.array(classes[k])
                m = np.mean(data,axis=0)
                std = np.std(data,axis=0)
                cov = covariance(data.transpose())
                self.cells[k] = {
                    "icov": np.linalg.inv(cov),
                    "cov_det": np.linalg.det(cov),
                    "mean": m,
                    "std": std,
                    "prob_priori": data.shape[0]/n
                }
            
    def __prob__(self,x,k):
        s = self.cells[k]["std"]
        m = self.cells[k]["mean"]
        ic = self.cells[k]["icov"]
        cov_det = self.cells[k]["cov_det"]
        prob_priori = self.cells[k]["prob_priori"]
        z = x-m
        return math.log(prob_priori) - 0.5*np.matmul(z, np.matmul(ic,z)) - 0.5*math.log(cov_det)
            
    def predict(self,x):
        distances = [(k,self.__prob__(x,k)) for k in self.cells]
        distances = sorted(distances,key=lambda x: x[1], reverse=True)
        return distances[0][0]
    
    
    def score(self,X,Y):
        predictions = [1 if self.predict(x)==y else 0 for x,y in zip(X,Y)] 
        return sum(predictions)/len(predictions)