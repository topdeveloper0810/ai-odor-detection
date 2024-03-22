import pandas as pd
import numpy as np
import os

# import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "train"
TEST_PATH = "test"

trainFilePath = os.path.abspath(TRAIN_PATH)
testFilePath = os.path.abspath(TEST_PATH)

# Train data
trainData = pd.read_csv(os.path.join(trainFilePath, "train.csv"))
trainData = trainData.iloc[:, :-2]
trainData["category"] = trainData["category"].astype("category")

# Test data
# testData = pd.read_csv(os.path.join(testFilePath, "test.csv"))
# testData = testData.iloc[:, :-2]

y = trainData["category"]
X = trainData.drop("category", axis=1)

# category = {"coffee": 0, "kahlua":1, "lrishCream":2, "rum":3}
# y_trans = y.map(category)
# print(y_trans)

# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()
stand = StandardScaler()

# ! Accuracy evaluation
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# X_train = stand.fit_transform(X_train)
# X_test = stand.transform(X_test)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print(y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: ", accuracy)

# ! Test Data
X = stand.fit_transform(X)
# testData = stand.transform(testData)
model.fit(X, y)
# y_pred = model.predict(testData)
# print(y_pred)

# ! CLI test odor
while True:
    odor = input("input : ")
    odor = odor.split(";")[1:-2]
    odor = np.array(odor)
    odor = pd.DataFrame([odor], columns=trainData.columns[1:])
    # print(testData)
    odor = stand.transform(odor)
    y_pred = model.predict(odor)
    print("Output : ", y_pred[0], "\n")
