import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("weather.csv")

print("Original Data:\n", data)

# 🔹 1. Handle Missing Values
data.fillna(method='ffill', inplace=True)

# 🔹 2. Remove Duplicates
data.drop_duplicates(inplace=True)

# 🔹 3. Convert categorical to numeric
le = LabelEncoder()

for col in data.columns:
    data[col] = le.fit_transform(data[col].astype(str))

# 🔹 4. Feature Scaling (optional)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# 🔹 5. Show cleaned data
print("\nPreprocessed Data:\n", data_scaled)

# 🔹 6. Save cleaned dataset
data_scaled.to_csv("weather_cleaned.csv", index=False)
