import numpy as np

class ConfusionMatrix():
    def __init__(self,label,predicted):
        self.matrix = np.array([[0,0],[0,0]])
        
        for l,p in zip(label,predicted):
            try:
                l = int(l)
                p = int(p)
                self.matrix[l][p] += 1
            except:
                raise Exception("Invalid value. Values must be 0 or 1")
                
    def true_positive(self):
        return self.matrix[1][1]
    
    def true_negative(self):
        return self.matrix[0][0]
    
    def false_positive(self):
        return self.matrix[0][1]
    
    def false_negative(self):
        return self.matrix[1][0]
    
    def sensitivity(self):
        if self.true_positive() == 0:
            return 0
        
        return self.true_positive()/(self.true_positive()+self.false_negative())
            
    def specificity(self):
        if self.true_negative() == 0:
            return 0
        
        return self.true_negative()/(self.true_negative()+self.false_positive())
    
    def precision(self):
        if self.true_positive() == 0:
            return 0
        
        return self.true_positive()/(self.true_positive()+self.false_positive())
        
    
    def accuracy(self):
        return (self.true_positive()+self.true_negative())/(np.sum(self.matrix))