import pandas as pd

# Extract
data = pd.read_csv("ds3.csv")

# Transform
data = data.dropna()
data = data.apply(lambda x: x.astype(str).str.lower())

# Load
data.to_csv("cleaned_data.csv", index=False)

print("ETL Process Done")
