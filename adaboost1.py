import numpy as np

class Adaboost():

    def __init__(self, no_of_clf=5):
        self.no_of_clf = no_of_clf

    def fit(self, x, y):
        no_of_samples, no_of_features = x.shape

        # Initial weights = 1/N
        w = np.full(no_of_samples, (1 / no_of_samples))

        self.clfs = []
        # Iterate through classifiers
        for _ in range(self.no_of_clf):
            clf = D_Stump()

            min_error = float('inf')

            # get best threshold and feature by greedy search
            for feature_i in range(no_of_features):
                X_column = x[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # polarity 1
                    p = 1
                    predictions = np.ones(no_of_samples)
                    predictions[X_column < threshold] = -1

                    # sum of weights of misclassified samples will be the error
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # save the best configuration
                    if error < min_error:
                        clf.pol = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            #print(clf.threshold, clf.feature_idx)

            # calculate performance, alpha
            # EPS = 1e-10
            EPS = 0
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # predictions
            predictions = clf.predict(x)
            # updated wrights
            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to 1
            w /= np.sum(w)

            # Append classifier
            self.clfs.append(clf)

    def predict(self, x):
        clf_preds = [clf.alpha * clf.predict(x) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred

# Weak classifier
class D_Stump():
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.pol = 1
        self.alpha = None

    def predict(self, x):
        n_samples = x.shape[0]
        x_column = x[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.pol == 1:
            predictions[x_column < self.threshold] = -1
        else:
            predictions[x_column > self.threshold] = -1

        return predictions