import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

# Import Data
spotifyData = pd.read_csv('data1.csv')

# Drop label and irrelevant classes
x = spotifyData.drop(['target','song_title', 'serial_num',  'mode', 'key', 'duration_ms'], axis=1)
y = spotifyData['target']

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# x[x.columns.values] = pd.DataFrame(x_scaled)

from sklearn.preprocessing import OneHotEncoder
#creating instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')

#perform one-hot encoding on 'team' column 
encoder_df = pd.DataFrame(encoder.fit_transform(x[['artist']]).toarray())

#merge one-hot encoded columns back with original DataFrame
x = x.join(encoder_df)

#drop 'artist' column
x.drop('artist', axis=1, inplace=True)


# x["artist"] = x["artist"].astype('category')
# x["artist"] = x["artist"].cat.codes
# x.head()

# print(x.dtypes)

#normalizing data
stddevs = np.std(x,axis=0)+1e-6
x /= stddevs
x /= stddevs


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)

# Fit model
svc = SVC(kernel='linear')
model = svc.fit(x_train, y_train)

# Test model
y_pred = model.predict(x_test)

# See results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))