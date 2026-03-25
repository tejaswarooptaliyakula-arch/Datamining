import pandas as pd
from scipy.spatial.distance import euclidean

data = pd.read_csv("ds3.csv")

# Statistics
print(data.describe())

# Similarity (example)
row1 = [1, 2, 3]
row2 = [2, 3, 4]

dist = euclidean(row1, row2)
print("Euclidean Distance:", dist)
