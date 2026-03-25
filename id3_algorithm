from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data = pd.read_csv("ds3.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X = pd.get_dummies(X)

model = DecisionTreeClassifier(criterion="entropy")  # ID3
model.fit(X, y)

print("Model trained using ID3")
