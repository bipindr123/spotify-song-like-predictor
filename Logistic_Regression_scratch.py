# Import Data
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

spotifyData = pd.read_csv('data1.csv')


# Drop label and irrelevant classes
x = spotifyData.drop(['target', 'song_title', 'serial_num'], axis=1)
y = spotifyData['target']
# print(len(x['artist'].unique()))
x["artist"] = x["artist"].astype('category')
x["artist"] = x["artist"].cat.codes
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x[x.columns.values] = pd.DataFrame(x_scaled)

# Create a PCA instance: pca
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# x = pd.DataFrame(principalComponents)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=69)

x_train =  x_train.to_numpy()
x_test =  x_test.to_numpy()
y_train =  y_train.to_numpy()
y_test =  y_test.to_numpy()

# a helper function to add a column of 1s to a matrix
def addones(X):
    return np.append(X,np.ones((X.shape[0],1)),axis=1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def ridgeregression(X,Y,lam):
    if lam==0.0:
        w = np.linalg.pinv(X)@Y
    else:
        myeye = np.eye(X.shape[1])
        myeye[-1,-1] = 0
        w = np.linalg.solve(X.T@X+lam*myeye,X.T@Y)
    return w

def learnlr(X,Y,lam=0.0):
    # X is an m-by-n matrix
    # Y is an m dimensional vector of 0s and 1s
    # lam is the regularization strength (scalar)
    # should return an (n+1) dimensional vector of weights
    
    ## Your code here
    newtrainX = addones(X)
    Y = np.where(Y>0,1,-1)
    #w = ridgeregression(newtrainX,Y,lam)
    w = np.zeros((newtrainX.shape[1],1))
    prev_loss  = np.sum(np.log(1+np.exp(-newtrainX@w*Y) ))
    eta = 1.0
    while eta >= 0.0000001:
        dL = 0.0
        for i in range(newtrainX.shape[0]):
            dL+= -(1-sigmoid(Y[i]*(w.T@newtrainX[i])))*(Y[i]*newtrainX[i])
        dL += 2*lam*w
        prev_w = w[:]
        w = w - eta*dL
        loss = 0.0
        loss  = np.sum(np.log(1+np.exp(-newtrainX@w*Y) ))
        for j in range(newtrainX.shape[1]):
            loss += lam * w[j]**2

        if(prev_loss-loss>=0.00001):
            prev_loss = loss
        else:
            w = prev_w
            eta*=0.7
    return w

w = learnlr(x_train,y_train)
new_x = addones(x_test)
wx = new_x@w
y_hat = np.where(wx>0,1,0)

acc = np.mean(y_hat==y_test)
print(acc)