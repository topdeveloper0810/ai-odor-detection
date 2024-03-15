import csv
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Setting
HOME_PATH = ""
DATESET_PATH = "datasets"
OUT_PATH = "out"
OUT_ZIP = "out.zip"

# Do not change these settings!
PREP_DROP = -1  # Drop a column
PREP_NONE = 0  # Perform no preprocessing on column of data
PREP_STD = 1  # Perform standardization on column of data
PREP_NORM = 2  # Perform normalization on column of data

### Read in .csv files to construct one long multi-axis, time series data

# Store header, raw data, and number of lines found in each .csv file
# header = None
numLines = []
fileNames = []

# Read each CSV file
for fileName in os.listdir(DATESET_PATH):
    rawData = []
    dataFrame = {}

    # Check if the path is a file
    filePath = os.path.abspath(DATESET_PATH + "/" + fileName)
    files = os.listdir(filePath)
    rowCount = len(files)
    colCount = 0
    if rowCount == 0:
        continue

    for index, file in enumerate(files):
        with open(filePath + "/" + file) as f:
            csvReader = csv.reader(f, delimiter=";")

            # validLineCounter = 0
            for lineCount, line in enumerate(csvReader):
                if lineCount == 0:
                    # rawData.append(line[1:] + [fileName])
                    rawData.append(line[1:])
                    if colCount == 0:
                        colCount = len(line) - 1
                    # validLineCounter += 1

                    # Record first header as our official header for all the data
                    # if header == None:
                    #     header = line

                    # Check to make sure subsequent headers match the original header
                    # if header == line:
                    # numLines.append(0)
                fileNames.append(fileName)
                # print(numLines, fileNames)
                # else:
                #     print("Error: Headers do not match. Skipping", fileName)
                #     break

    # rawData.insert(0, np.linspace(1, 66))

    # Convert our raw data into a numpy array
    rawData = np.array(rawData).astype(float)

    for j in range(colCount):
        colData = []
        for i in range(rowCount):
            colData.append(rawData[i][j])
        header = f"{j + 1}"
        dataFrame[header] = colData

    df = pd.DataFrame(dataFrame)
    df.to_excel(f"{fileName}.xlsx")

    header = df.columns.tolist()
    # print(header)

    ### Analyze the data

    # Calculate means, standard deviations, and ranges
    means = np.mean(df, axis=0)
    stdDevs = np.std(df, axis=0)
    maxes = np.max(df, axis=0)
    mins = np.min(df, axis=0)
    ranges = np.ptp(df, axis=0)

    # Print results
    # for i, name in enumerate(header):
    #     print(name)
    #     print("  mean:", means[i])
    #     print("  std dev:", stdDevs[i])
    #     print("  max:", maxes[i])
    #     print("  min:", mins[i])
    #     print("  range:", ranges[i])

    ### Choose preprocessing method for each column
    # PREP_DROP: Drop column
    # PREP_NONE: no preprocessing
    # PREP_STD: standardization (if data is Gaussian)
    # PREP_NORM: normalization (if data is non-Gaussian)

    # Change this to match ur picks!
    preproc = [PREP_NORM for _ in range(len(header))]
    # print(preproc)

    # Check to make sure we have the correct number of preprocessing request elements
    assert(len(preproc) == len(header))
    # assert(len(preproc) == len(df.shape[1]))

    ### Perform preprocessing steps as requested

    # Figure out how many columns we plan to keep
    numCols = sum(1 for x in preproc if x != PREP_DROP)

    # Create empty numpy array and header for preprocessed data
    prepData = np.zeros((df.shape[0], numCols))
    prepHeader = []
    prepMeans = []
    prepStdDevs = []
    prepMins = []
    prepRanges =[]

    # Go through each column to preprocess the data
    prepC = 0
    for rawC in range(len(header)):

        # Drop column if requested
        if preproc[rawC] == PREP_DROP:
            print("Dropping", header[rawC])
            continue

        # Perform data standardization
        if preproc[rawC] == PREP_STD:
            prepData[:, prepC] = (df.iloc[:, rawC] - means[rawC]) / stdDevs[rawC]

        # Perform data normalization
        elif preproc[rawC] == PREP_NORM:
            prepData[:, prepC] = (df.iloc[:, rawC] - mins[rawC]) / ranges[rawC]

        # Copy data over if no preprocessing is requested
        elif preproc[rawC] == PREP_NONE:
            prepData[:, rawC] = df[:, rawC]

        # Error if code not recogized
        else:
            raise Exception("Preprocessing code not recognized")

        # Copy header (and preprocessing constants) and increment preprocessing column index
        prepHeader.append(header[rawC])
        prepMeans.append(means[rawC])
        prepStdDevs.append(stdDevs[rawC])
        prepMins.append(mins[rawC])
        prepRanges.append(ranges[rawC])
        prepC += 1

    # Preprocessed data
    prepData = np.array(prepData).astype(float)
    outDf = pd.DataFrame(prepData)
    outDf.to_excel(f"{fileName}.xlsx")
    # print(prepData)
    
    ### Zip output directory
    # %cd {OUT_PATH}
    # !zip -FS -r -q {OUT_PATH} *
    # %cd {HOME_PATH}
