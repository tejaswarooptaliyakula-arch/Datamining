from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv("ds3.csv")

X = pd.get_dummies(data)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(kmeans.labels_)
