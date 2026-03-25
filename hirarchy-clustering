from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("ds3.csv")

X = pd.get_dummies(data)

Z = linkage(X, method='ward')

dendrogram(Z)
plt.show()
