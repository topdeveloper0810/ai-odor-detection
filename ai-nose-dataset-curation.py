import csv
import os
import numpy as np
import pandas as pd

### Setting
HOME_PATH = ""
DATESET_PATH = "datasets"

### Read in .csv files to construct one long multi-axis, time series data

# Store header, raw data, and number of lines found in each .csv file
# header = None
rawData = []
numLines = []
fileNames = []
dataFrame = {}

# Read each CSV file
for fileName in os.listdir(DATESET_PATH):

    # Check if the path is a file
    filePath = os.path.abspath(DATESET_PATH + "/" + fileName)
    files = os.listdir(filePath)
    if len(files) == 0:
        continue

    for file in files:
        with open(filePath + "/" + file) as f:
            csvReader = csv.reader(f, delimiter=";")

            validLineCounter = 0
            for lineCount, line in enumerate(csvReader):
                # if lineCount == 0:
                #     rawData.append(["category"] + [f"sensor{j}" for j in range(1, 67)])
                rawData.append([fileName] + line[1:])
# rawData = np.array(rawData).astype(float)
rawData = np.array(rawData)

# Print out our results
print("Dataset array shape:", rawData.shape)

for j in range(rawData.shape[1]):
    colData = []
    for i in range(rawData.shape[0]):
        colData.append(rawData[i][j])
    if j == 0:
        header = "category"
    else:
        header = f"sensor{j}"
    dataFrame[header] = colData

df = pd.DataFrame(dataFrame)
df.to_csv("total.csv")

### First Train dataset is created.