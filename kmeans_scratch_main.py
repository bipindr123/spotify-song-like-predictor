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
from sklearn.metrics import accuracy_score


def RunKMeans(random_state):
    spotifyData = pd.read_csv('data1.csv')
    # Drop label and irrelevant classes
    x = spotifyData.drop(['target', 'song_title', 'serial_num'], axis=1)
    y = spotifyData['target']

    x["artist"] = x["artist"].astype('category')
    x["artist"] = x["artist"].cat.codes
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x[x.columns.values] = pd.DataFrame(x_scaled)

    def pca(X, k):
        # center the data
        X_meaned = X - np.mean(X, axis=0)

        # calculate covariance matrix
        cov_mat = np.cov(X_meaned, rowvar=False)

        # get eigen vectors and values
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)

        # sort them
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        # select the first highest given k componenets
        eigenvector_subset = sorted_eigenvectors[:, 0:k]
        X_reduced = np.dot(eigenvector_subset.transpose(),
                        X_meaned.transpose()).transpose()
        return X_reduced
    #using PCA to reduce the dimentions to only 2 columns
    x = pd.DataFrame(pca(x, 2))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state)

    # X = x_test.values
    # sns.scatterplot(X[:, 0], X[:, 1])
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.show()


    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    class MyKmeans:
        def __init__(self, k=2, threshold=0.001, max_iterations=10000):
            self.k = k
            self.threshold = threshold
            self.max_iterations = max_iterations
        
        def predict(self, X):
            distances = [np.linalg.norm(X - self.centroids[centroid]) for centroid in self.centroids]
            label = distances.index(min(distances))
            return label

        def fit(self, X):

            self.centroids = {}

            #initialize centroids randomly
            rng = np.random.default_rng()
            rngs = rng.choice(X.shape[0],self.k,replace=False)
            for i,random_num in enumerate(rngs):
                self.centroids[i] = X[random_num]

            for i in range(self.max_iterations):
                self.classes = {}

                for i in range(self.k):
                    self.classes[i] = []

                #calculate difference between the features and centroid
                for row in X:
                    distances = [np.linalg.norm(row - self.centroids[centroid]) for centroid in self.centroids]
                    label = distances.index(min(distances))
                    self.classes[label].append(row)

                #deep copy of centroids
                prev_centroids = self.centroids.copy()

                #take avarage of values
                for label in self.classes:
                    self.centroids[label] = np.average(self.classes[label], axis=0)

                #breaks if the difference of previous and current centroids percent is lesser than threshold
                for c in self.centroids:
                    prev_center = prev_centroids[c]
                    current_centroid = self.centroids[c]
                    if np.sum((current_centroid-prev_center)/prev_center*100.0) < self.threshold:
                        return

    #training the data
    my_kmeans = MyKmeans()
    my_kmeans.fit(x_train)
    
    #testing the data
    y_pred = []
    for i in range(len(x_test)):
        predict_me = x_test[i][:]
        predict_me = predict_me.reshape(-1, len(predict_me))
        y_pred.append(my_kmeans.predict(predict_me))
    y_pred = np.array(y_pred)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # sklearn implementation
    kmeans = KMeans(n_clusters=2)
    model = kmeans.fit(x_train)
    # Test model
    y_hat = model.predict(x_test)
    colors = 10*["g","r","c","b","k"]
    # See results
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    for centroid in my_kmeans.centroids:
        plt.scatter(my_kmeans.centroids[centroid][0], my_kmeans.centroids[centroid][1],
                    marker="o", color="k", s=150, linewidths=5)

    for classification in my_kmeans.classes:
        color = colors[classification]
        for featureset in my_kmeans.classes[classification]:
            plt.scatter(featureset[0], featureset[1], marker=".", color=color, s=50, linewidths=5)
    
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Logistic Regression Confusion Matrix', fontsize=18)
    plt.show()

    plt.show()
    return accuracy_score(y_test, y_pred) , accuracy_score(y_test, y_hat)

# RunKMeans(2)