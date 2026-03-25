# KDD Process Example

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Data Selection
data = pd.read_csv("ds3.csv")

# Step 2: Data Cleaning
data = data.dropna()

# Step 3: Transformation
le = LabelEncoder()
for col in data.columns:
    data[col] = le.fit_transform(data[col].astype(str))

print(data.head())
