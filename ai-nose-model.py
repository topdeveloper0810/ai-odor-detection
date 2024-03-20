import pandas as pd

# import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("total.csv")
data = data.iloc[:, 1:]
data = data.iloc[:, :-2]
# print(data.head())

data["category"] = data["category"].astype("category")

# test = pd.read_csv("test.csv")
# test = test.iloc[:, 1:]
# test = test.iloc[:, :-2]

# print(test_scaled)

y = data["category"]
X = data.drop("category", axis=1)
# print(X.head())

# category = {"coffee": 0, "kahlua":1, "lrishCream":2, "rum":3}
# y_trans = y.map(category)
# print(y_trans)

# clf = LogisticRegression()
# clf = DecisionTreeClassifier()
clf = RandomForestClassifier()

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

stand = StandardScaler()
X_train = stand.fit_transform(X)
# X_test = stand.transform(test)
clf.fit(X_train, y)
# clf.fit(X_scaled, y)
# print(X_scaled)

while True:
    test = input(">>>input : ")
    test = test.split(";")[1:-2]
    # print(test)
    testData = pd.DataFrame([test], columns=X.columns)
    # print(testData)
    testData = stand.transform(testData)
    y_pred = clf.predict(testData)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(y_pred)
    print(">>>>Output : ", y_pred[0])
    # print("Accuracy:", accuracy)
