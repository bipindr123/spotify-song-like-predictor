import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Import Data
spotifyData = pd.read_csv('data1.csv')

# Drop label and irrelevant classes
x = spotifyData.drop(['target','song_title', 'artist'], axis=1)
y = spotifyData['target']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=69)

# Fit model
rfc = RandomForestClassifier(n_estimators = 100)
model = rfc.fit(x_train, y_train)

# Test model
y_pred = model.predict(x_test)

# See results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))