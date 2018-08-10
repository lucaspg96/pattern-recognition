import abc

class LearningAlgorithm():
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def fit(self,X,Y,itter=100):
        return
    
    @abc.abstractmethod
    def predict(self,X):
        return
    
    @abc.abstractmethod
    def score(self,X,Y):
        return