import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold



df = pd.read_csv('winequality-white.csv', sep=';')

count_unique = df['quality'].value_counts()
print('Unique quality', count_unique)

# Histogram for quality
df['quality'].hist()

plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()


# Data splitting - Baseline
X = df.drop(columns=['quality'])
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rkf = RepeatedKFold(n_splits=3, n_repeats=10)
rkf.get_n_splits()

# Metric

