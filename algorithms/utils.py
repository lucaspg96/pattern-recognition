import numpy as np
import math
import matplotlib.pyplot as plt
from lxml import etree as et
import pandas as pd

def manhattan_distance(a,b):
    return sum(a-b)

def euclidian_distance(a,b):
    return np.linalg.norm(a-b)

def train_test_split(data,train_rate, shuffle=True):
    
    if shuffle:
        np.random.shuffle(data)
    
    train_size = math.floor(train_rate*data.shape[0])

    train_data = data[0: train_size,:]
    test_data = data[train_size:,:]
    
    return train_data,test_data

def covariance(X):
    m = np.mean(X,axis=1).transpose()
    n = X.shape[1]
    R = np.matmul(X,X.transpose())
    return R - np.outer(m,m.transpose())

def generate_mahalanobis_distance(Q):
    def distance_function(x,y):
        z = x - y
        right = np.matmul(Q,z)
        left = np.matmul(z.transpose(),right)
        return math.sqrt(left)
    
    return distance_function

def pooled_covariance(clusters):
    sigma = None
    n = sum([len(clusters[k]) for k in clusters])
    
    for k in clusters:
        data = np.array(clusters[k])
        cov = covariance(data.transpose())
        if sigma is None:
            sigma = (cov.shape[0]/n)*cov
        else:
            sigma += (cov.shape[0]/n)*cov
        
    return sigma

def friedman_regularization(l,g,clusters):
    sk = {}
    
    wk = {}
    
    cov_lambda = {}
    
    for k in clusters:
        data = np.array(clusters[k])
        wk[k] = data.shape[0]
        sk[k] = covariance(data.transpose())*wk[k]
    
    s = sum([sk[k] for k in sk])
    w = sum([wk[k] for k in wk])
    
    for k in clusters:
        wk_lambda = (1-l)*wk[k] + l*w
        sk_lambda = (1-l)*sk[k] + l*s
        cov_lambda[k] = (1/wk_lambda)*sk_lambda
        
    return {
        k: (1-g)*cov_lambda[k] + (g/cov_lambda[k].shape[0])*np.trace(cov_lambda[k])*np.eye(cov_lambda[k].shape[0],cov_lambda[k].shape[0]) \
        for k in clusters
    }

def is_invertible(X):
    rank = np.linalg.matrix_rank(X)
    cond = np.linalg.cond(X)
    if rank < X.shape[0]:
        return False, "Matrix isn't inversible because has rank {}".format(rank)
    if cond > 30:
        return False, "Matrix is ill-conditioned({})".format(cond)
    return True

def one_hot_encode(x):
    n = max(x)+1
    matrix = np.zeros((len(x),n))
    for i,c in enumerate(x):
        matrix[i,c] = 1
        
    return matrix

def load_xml_news(path,plot=False):
    parser = et.XMLParser(recover=True)
    data = et.parse(path, parser=parser).getroot()

    tags = ["title","link","pubDate","description","image","text","when"]
    hist = {}
    for t in tags:
        hist[t] = 0;

    items = data.findall("item")
    total = len(items)
    news = []
    labels = []

    for c in items:
        aux = ""
        text = c.find("text").text
        desc = c.find("description").text
        if not text == None:
            aux += text
        if not desc == None:
            aux += desc
        if aux!="":
            news.append(aux)
            labels.append(c.get("category"))

        for t in tags:
            if c.find(t).text != None:
                hist[t] += 1
                
    if plot:
        df = pd.DataFrame({"category":labels})
        fig = plt.figure(figsize=(10,6))
        plt.title("Ocorrências das categorias")
        _ = df["category"].value_counts().plot(kind="bar")
        
    return news, labels

def load_articles(path, delimiter="\t", plot=False):
    df = pd.read_csv(path,delimiter=delimiter)
    
    news = []
    labels = []

    for row in df.iterrows():
        row = row[1]
        news.append(row.get("text"))
        labels.append(row.get("category"))

    if plot:
        _ = df["category"].value_counts().plot(kind="bar")
        
    return news, labels   