import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


df = pd.read_csv('winequality-white.csv', sep=';')

count_unique = df['quality'].value_counts()


print('Unique quality', count_unique)

percent_of_total = (count_unique / df.shape[0]) * 100

print(percent_of_total)


# Barchart for quality
plt.bar(count_unique.index, count_unique.values)
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()


# Data splitting - Baseline
X = df.drop(columns=['quality'])
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# Logistic Regression classifier (lbfgs)
rkf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
rkf.get_n_splits()
pipe = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(solver='lbfgs', max_iter=2000))])
scores = cross_val_score(pipe, X_train, y_train, cv=rkf, scoring='f1_macro')


print(f"Cross-validation accuracy scores: {scores}")
print(f"Mean CV accuracy: {np.mean(scores):.4f}")
print(f"Standard deviation of CV accuracy: {np.std(scores):.4f}")


# Random forest classifier
pipe2 = Pipeline([('scalar', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))])
scores = cross_val_score(pipe2, X_train, y_train, cv=rkf, scoring='f1_macro')


print(f"Cross-validation accuracy scores: {scores}")
print(f"Mean CV accuracy: {np.mean(scores):.4f}")
print(f"Standard deviation of CV accuracy: {np.std(scores):.4f}")




clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)

y_pred = clf.predict(X_test)

f1_macro = f1_score(y_test, y_pred, average="macro")
print(f1_macro)