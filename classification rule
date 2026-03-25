from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("ds3.csv")

X = pd.get_dummies(data.iloc[:, :-1])
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()  # J48 equivalent
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
