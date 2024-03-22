import csv
import os
import numpy as np
import pandas as pd

### Setting
HOME_PATH = ""
DATESET_PATH = "dataset"
TRAIN_PATH = "train"
TEST_PATH = "test"

### Read in .csv files to construct one long multi-axis, time series data

# Store header, raw data, and number of lines found in each .csv file
# header = None
trainData = []
testData = []
folderNames = []
dataFrame = {}

# Read each CSV file
for folderName in os.listdir(DATESET_PATH):
    # Check if the path is a file
    filePath = os.path.abspath(DATESET_PATH + "/" + folderName)
    files = os.listdir(filePath)
    if len(files) == 0:
        continue

    for file in files:

        category = file.split(".")[0]
        with open(filePath + "/" + file) as f:
            csvReader = csv.reader(f, delimiter=";")

            for lineCount, line in enumerate(csvReader):
                if category == "test":
                    testData.append(line[1:])
                else:
                    trainData.append([category] + line[1:])
                # print(rawData)
# rawData = np.array(rawData).astype(float)
trainData = np.array(trainData)
testData = np.array(testData)

# Print out our results
# print("Dataset array shape:", trainData.shape)
# print("Dataset array shape:", testData.shape)

# Train Dataset
dataFrame = {}
for j in range(trainData.shape[1]):
    colData = []
    for i in range(trainData.shape[0]):
        colData.append(trainData[i][j])
    if j == 0:
        header = "category"
    else:
        header = f"sensor{j}"
    dataFrame[header] = colData

trainDf = pd.DataFrame(dataFrame)
trainFilePath = os.path.abspath(TRAIN_PATH)
if not os.path.exists(trainFilePath):
    os.makedirs(trainFilePath)

trainDf.to_csv(os.path.join(trainFilePath, "train.csv"), index=False)
print("train data is ready.")

# Test Dataset
dataFrame = {}
for j in range(testData.shape[1]):
    colData = []
    for i in range(testData.shape[0]):
        colData.append(testData[i][j])
    header = f"sensor{j+1}"
    dataFrame[header] = colData

testDf = pd.DataFrame(dataFrame)
testFilePath = os.path.abspath(TEST_PATH)
if not os.path.exists(testFilePath):
    os.makedirs(testFilePath)

testDf.to_csv(os.path.join(testFilePath, "test.csv"), index=False)
print("train data is ready.")
