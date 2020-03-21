from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import time
import pandas as pd
import warnings
from sklearn import decomposition
from sklearn.decomposition import FastICA
from sklearn.metrics import silhouette_samples,silhouette_score,adjusted_mutual_info_score
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture


df = pd.read_csv('digits_training.tra', sep=",", skiprows=0)

df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:64]
tar1=df[:,64]
X=dat
y=tar1

r=np.array([7,17,37,57,77])
plt.figure()
for m in range(5):
    x1=[]
    a1=[]
    a2=[]
    t1=[]
    t2=[]
    i1=[]
    for i in range(1,32,3):
        rp1 = GaussianRandomProjection(n_components=i,random_state=r[m])
        rp1.fit(X)
        X1 = rp1.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=20)
        clf = MLPClassifier(solver='sgd',activation='relu',alpha=0.03,momentum=0.9,tol=0.001,learning_rate='adaptive')
        s1 = time.time()
        clf.fit(X_train, y_train)
        e1 = time.time()
        s2 = time.time()
        y_prd = clf.predict(X_test)
        e2 = time.time()
    
        t1.append(e1-s1)
        t2.append(e2-s2)
    
        y_test = np.array(y_test)
        y_prd = np.array(y_prd)
        a2.append(clf.best_loss_)
        a1.append(accuracy_score(y_test, y_prd))
        x1.append(i)
        i1.append(clf.n_iter_)
        print((clf.best_loss_),clf.n_iter_, clf.n_layers_, clf.n_outputs_)
        print('Test Accuracy: %.8f' % accuracy_score(y_test, y_prd))
        print("Training time: ", e1-s1)
        print("Testing time: ", e2-s2)


    i1=np.array(i1)
    t2=np.array(t2)
    t1=np.array(t1)
    a1=np.array(a1)
    a2=np.array(a2)
    x1=np.array(x1)
    
    
    plt.subplot(2,2,1)
    if m == 1:
        plt.plot(x1, a1, color='r', label=1)
    if m == 2:
        plt.plot(x1, a1, color='g', label=2)
    if m == 3:
        plt.plot(x1, a1, color='b', label=3)
    if m == 4:
        plt.plot(x1, a1, color='y', label=4)
    if m == 0:
        plt.plot(x1, a1, color='c', label=0)
    plt.xlabel('number of components')
    plt.ylabel('scores')
    plt.title(' test acc vs # of components')
    plt.legend()
    
    plt.subplot(2,2,2)
    if m == 1:
        plt.plot(x1, i1, color='r', label=1)
    if m == 2:
        plt.plot(x1, i1, color='g', label=2)
    if m == 3:
        plt.plot(x1, i1, color='b', label=3)
    if m == 4:
        plt.plot(x1, i1, color='y', label=4)
    if m == 0:
        plt.plot(x1, i1, color='c', label=0)
    plt.xlabel('number of components')
    plt.ylabel('# of iterations')
    plt.title(' # of iterations vs #components')
    plt.legend()
    
    plt.subplot(2,2,3)
    if m == 1:
        plt.plot(x1, t1, color='r', label=1)
    if m == 2:
        plt.plot(x1, t1, color='g', label=2)
    if m == 3:
        plt.plot(x1, t1, color='b', label=3)
    if m == 4:
        plt.plot(x1, t1, color='y', label=4)
    if m == 0:
        plt.plot(x1, t1, color='c', label=0)
    plt.xlabel('number of components')
    plt.ylabel('training time')
    plt.title('training time vs # of components')
    plt.legend()
    
    
    plt.subplot(2,2,4)
    if m == 1:
        plt.plot(x1, a2, color='r', label=1)
    if m == 2:
        plt.plot(x1, a2, color='g', label=2)
    if m == 3:
        plt.plot(x1, a2, color='b', label=3)
    if m == 4:
        plt.plot(x1, a2, color='y', label=4)
    if m == 0:
        plt.plot(x1, a2, color='c', label=0)
    plt.xlabel('number of components')
    plt.ylabel('best training loss')
    plt.title(' best training loss vs # of components ')
    plt.legend()
plt.suptitle('Neural network performance with RP on digits dataset')
plt.subplots_adjust(hspace=0.45)
plt.show()