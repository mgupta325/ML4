import numpy as np
import scipy
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import pinv
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


df = pd.read_csv('tic-tac-toe.data', sep=",", skiprows=0)

df=np.array(df)
print(df.shape,df.dtype)
dat=df[:,0:9]
tar1=df[:,9]
X=dat
y=tar1

rp1 = GaussianRandomProjection(n_components=2)
X1 = rp1.fit_transform(X)
plt.figure()
for i in range(len(y)):
    if y[i] == 0:
        plt.scatter(X1[i, 0], X1[i, 1], color='r')
    elif y[i] == 1:
        plt.scatter(X1[i, 0], X1[i, 1], color='b')
plt.title('visualization of data in 2D (rp)-> tic-tac toe dataset')
plt.show()

r=np.array([7,17,37,57,77])
plt.figure()
for m in range(5):


    x1=[]
    # e1=[]
    r1=[]
    for i in range(1, 10, 2):
        rp1 = GaussianRandomProjection(n_components=i,random_state=r[m])
        rp1.fit(X)
        X1 = rp1.transform(X)
        print(X1.shape,rp1.components_.shape)
        X2 = np.dot(X1,(rp1.components_))
        rmse = np.sqrt(mean_squared_error(X, X2))
        x1.append(i)
        r1.append(rmse)

    r1 = np.array(r1)
    x1 = np.array(x1)

    print(r1)


    if m==1:
        plt.plot(x1, r1,color='r',label=1)
    if m==2:
        plt.plot(x1, r1,color='g',label=2)
    if m==3:
        plt.plot(x1, r1,color='b',label=3)
    if m==4:
        plt.plot(x1, r1,color='y',label=4)
    if m==0:
        plt.plot(x1, r1,color='c',label=0)

plt.xlabel('number of components')
plt.ylabel('error')
plt.title('reconstruction error of tic tac toe dataset with different random seeds in rp')
plt.legend()
plt.show()

###############
rp=GaussianRandomProjection(n_components=3,random_state=77)
X12=rp.fit_transform(X)




sse = {}
# elbow method
for k in range(2, 20, 4):
    kmeans = KMeans(n_clusters=k, max_iter=500).fit(X12)
    # print(kmeans.labels_)
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title("tic tac toe dataset clustering")
plt.show()

# silhouette analysis, checking for lining up of clusters with actual labels(mutual_info_score)
range_n_clusters = [2, 6,10,12]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(12, 5)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X12) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X12)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X12, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    print(n_clusters, adjusted_mutual_info_score(y, clusterer.labels_))
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X12, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on tic tac toe dataset "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

##em

lowest_bic = np.infty
bic = []
n_components_range = range(2, 14,2)
cv_types = ['spherical', 'tied', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type, random_state=777)
        gmm.fit(X12)
        gmm_labels = gmm.predict(X12)
        bic.append(gmm.bic(X12))
        if abs(bic[-1]) < lowest_bic:
            lowest_bic = abs(bic[-1])
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model for tic tac toe dataset')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
       .2 * np.floor(bic.argmin() / len(n_components_range))

spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)
print(clf.covariances_.shape, clf.covariance_type, clf.n_components)
# clf.covariance_type='full'
# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X12)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X12[Y_ == i, 0], X12[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model components = %d' % clf.n_components)
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()

range_n_clusters = [2, 6,10,12]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(12, 5)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X12) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    gmm = mixture.GaussianMixture(n_components=n_clusters,
                                  covariance_type='full', random_state=777)
    gmm.fit(X12)
    cluster_labels = gmm.predict(X12)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X12, cluster_labels)
    print("For n_components =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    print(n_clusters, adjusted_mutual_info_score(y, cluster_labels))
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X12, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various components.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Component label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for EM on tic tac toe dataset "
                  "with n_components = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

