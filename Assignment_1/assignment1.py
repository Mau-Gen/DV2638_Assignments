import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error



df = pd.read_csv('winequality-white.csv', sep=';')

count_unique = df['quality'].value_counts()
print('Unique quality', count_unique)

percent_of_total = (count_unique / df.shape[0]) * 100

print(percent_of_total)

# Histogram for quality
df['quality'].hist()

plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()


# Data splitting - Baseline
X = df.drop(columns=['quality'])
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



rkf = RepeatedKFold(n_splits=3, n_repeats=10)
rkf.get_n_splits()

scores = [] # To store r-squared scores

rmse_list = [] # To store root mean squared error 

lrmodel = LinearRegression()

for train_index, test_index in rkf.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    lrmodel.fit(X_train, y_train)
    y_predict = lrmodel.predict(X_test)
    scores.append(lrmodel.score(X_test, y_test))
    rmse_fold = root_mean_squared_error(y_test, y_predict)
    rmse_list.append(rmse_fold)

print(f"R Squared Mean: {np.mean(scores)}")
print(f"RMSE Mean: {np.mean(rmse_list)}")

# Metric

