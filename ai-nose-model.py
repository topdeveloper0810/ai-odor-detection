import pandas as pd

# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("total.csv")
data = data.iloc[:, 1:]

data["category"] = data["category"].astype("category")

# test = pd.read_csv("test.csv")
# test = test.iloc[:, 1:]

y = data["category"]
X = data.drop("category", axis=1)

# clf = DecisionTreeClassifier()
clf = RandomForestClassifier()
# clf = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=42
# )
# clf.fit(X_train, y_train)
clf.fit(X, y)

while True:
    test = input("input : ")
    test = test.split(";")[1:]
    # print(test)
    testData = pd.DataFrame([test], columns=X.columns)
    # print(testData)
    y_pred = clf.predict(testData)
    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(y_pred)
    print("Output : ", y_pred[0])
    # print("Accuracy:", accuracy)
