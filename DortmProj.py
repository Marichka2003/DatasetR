import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = make_blobs(n_samples = 50, n_features = 2, centers = 4,cluster_std = 1.5, random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
k_table = []
accuracy_table = []
for k in np.arange(5,27,4):
    regression = KNeighborsClassifier(n_neighbors = k)
    regression.fit(X_train, y_train)
    y_predict = regression.predict(X_test)
    k_table.append(k)
    accuracy_table.append(accuracy_score(y_test, y_predict)*100)
result_table = {'K': k_table, 'Accuracy': accuracy_table }
result_df = pd.DataFrame(result_table)
print(result_df)
print("The best accuracy = ", result_df['Accuracy'].max())
for i in range (len(result_df)):
    if(result_df.iloc[i]['Accuracy'] == max(accuracy_table)):
        print("The best K = ", result_df.iloc[i]['K'])
        break
