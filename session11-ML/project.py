import pandas as pd
data = pd.read_csv("session11-ML/data.csv")

print(data)

print(data.isnull())

data["sleep"] = data["sleep"].fillna(data["sleep"].mean())
data["exercise"] = data["exercise"].fillna(data["exercise"].mean())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data[["study", "sleep", "exercise"]]
y = data["pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

predict = model.predict(X_test)
for i in range(len(X_test)):
    print(X_test.iloc[i].values, "-> ", predict[i], "| real: ", y_test.iloc[i])

print(f"Accuracy: {accuracy_score(y_test, predict)}")
