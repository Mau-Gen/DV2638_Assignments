import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('winequality-white.csv', sep=';')
print(df.describe())

# Histogram for quality
df['quality'].hist()

plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()


