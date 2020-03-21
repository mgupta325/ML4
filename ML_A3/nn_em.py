import numpy as np
import itertools
import time
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import mixture
from sklearn.metrics import silhouette_samples,silhouette_score,adjusted_mutual_info_score
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('digits_training.tra', sep=",", skiprows=0)

df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:64]
tar1=df[:,64]
X=dat
y=tar1

x1 = []
a1 = []
a2 = []
t1 = []
t2 = []
i1 = []

for k in range(2, 16, 2):
    gmm = mixture.GaussianMixture(n_components=k,
                                  covariance_type='full', random_state=777)
    gmm.fit(X)
    cluster_labels = gmm.predict(X)
    l1 = np.reshape(cluster_labels, (5619, 1))
    print(cluster_labels.shape, l1.shape, X.shape)
    print(cluster_labels)
    X1 = np.hstack((l1, X))
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=20)
    clf = MLPClassifier(solver='sgd', activation='relu', alpha=0.03, momentum=0.9, tol=0.001, learning_rate='adaptive')
    s1 = time.time()
    clf.fit(X_train, y_train)
    e1 = time.time()
    s2 = time.time()
    y_prd = clf.predict(X_test)
    e2 = time.time()

    t1.append(e1 - s1)
    t2.append(e2 - s2)

    y_test = np.array(y_test)
    y_prd = np.array(y_prd)
    a2.append(clf.best_loss_)
    a1.append(accuracy_score(y_test, y_prd))
    x1.append(k)
    i1.append(clf.n_iter_)
    print((clf.best_loss_), clf.n_iter_, clf.n_layers_, clf.n_outputs_)
    print('Test Accuracy: %.8f' % accuracy_score(y_test, y_prd))
    print("Training time: ", e1 - s1)
    print("Testing time: ", e2 - s2)

i1 = np.array(i1)
t2 = np.array(t2)
t1 = np.array(t1)
a1 = np.array(a1)
a2 = np.array(a2)
x1 = np.array(x1)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x1, a1)
plt.xlabel('number of clusters')
plt.ylabel('scores')
plt.ylim(0.85,1.1)
plt.title(' test acc vs # of clusters')

plt.subplot(2, 2, 2)
plt.plot(x1, i1)
plt.xlabel('number of clusters')
plt.ylabel('# of iterations')
plt.title(' # of iterations vs #clusters')

plt.subplot(2, 2, 3)
plt.plot(x1, t1)
plt.xlabel('number of clusters')
plt.ylabel('training time')
plt.title('training time vs # of clusters')

plt.subplot(2, 2, 4)
plt.plot(x1, a2)
plt.xlabel('number of clusters')
plt.ylabel('best training loss')
plt.title(' best training loss vs # of clusters ')
plt.suptitle('Neural network performance with EM on digits dataset')
plt.subplots_adjust(hspace=0.45)
plt.show()

