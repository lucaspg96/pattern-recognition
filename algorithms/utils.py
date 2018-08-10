import numpy as np
import math

def manhattan_distance(a,b):
    return sum(a-b)

def euclidian_distance(a,b):
    return np.linalg.norm(a-b)

def train_test_split(data,train_rate):
    train_size = math.floor(train_rate*data.shape[0])

    train_data = data[0: train_size,:]
    test_data = data[train_size:,:]
    
    return train_data,test_data