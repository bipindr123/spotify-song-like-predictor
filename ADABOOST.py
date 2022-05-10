import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Import Data
spotifyData = pd.read_csv('data1.csv')
spotifyData = spotifyData.sample(frac=1).reset_index(drop=True)

# Drop label and irrelevant classes
x = spotifyData.drop(['target','song_title', 'artist', 'serial_num'], axis=1)
y = spotifyData['target']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#create classifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=0.5)

#fit model
model = abc.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

# See results
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report: ")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score: ")
print(accuracy_score(y_test, y_pred))
print("\nROC AUC Score: ")
print(roc_auc_score(y_test, y_pred))