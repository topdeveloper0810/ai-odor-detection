import csv
import os
import numpy as np
import pandas as pd

### Setting
HOME_PATH = ""
DATESET_PATH = "testsets"

### Read in .csv files to construct one long multi-axis, time series data

# Store header, raw data, and number of lines found in each .csv file
# header = None
rawData = []
numLines = []
fileNames = []
dataFrame = {}

# Check if the path is a file
filePath = os.path.abspath(DATESET_PATH)
files = os.listdir(filePath)
# print(files)
if len(files) == 0:
    print("Testset not found.")
else:
    for file in files:
        with open(filePath + "/" + file) as f:
            csvReader = csv.reader(f, delimiter=";")
            for lineCount, line in enumerate(csvReader):
                rawData.append(line[1:])
    # rawData = np.array(rawData).astype(float)
    rawData = np.array(rawData)

    # Print out our results
    print("Dataset array shape:", rawData.shape)

    for j in range(rawData.shape[1]):
        colData = []
        for i in range(rawData.shape[0]):
            colData.append(rawData[i][j])

        header = f"sensor{j+1}"
        dataFrame[header] = colData

    df = pd.DataFrame(dataFrame)
    df.to_csv("test.csv")

    ### Second, Test dataset is prepaired.