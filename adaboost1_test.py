from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
#from sklearn import preprocessing
#from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from adaboost1 import Adaboost

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

data = spotifyData = pd.read_csv('data1.csv')
x = data.drop(['target', 'serial_num'], axis=1)
#x = data[['instrumentalness', 'loudness', 'energy']]
y = data['target']


x = x.to_numpy()
y = y.to_numpy()

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 5)

clf = Adaboost(no_of_clf=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc =accuracy(y_test, y_pred)
print("Accuracy: ", acc)

print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nROC AUC Score: ")
print(roc_auc_score(y_test, y_pred))

# conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
# #
# # Print the confusion matrix using Matplotlib

# fig, ax = plt.subplots(figsize=(5.0, 5.0))
# ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('AdaBoost Confusion Matrix', fontsize=18)
# plt.show()

# ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('AdaBoost ROC Curve', fontsize=18)
plt.show()