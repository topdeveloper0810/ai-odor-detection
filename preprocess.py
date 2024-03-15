import os
import pandas as pd

# Specify the folder paths
input_folder = 'Nano-data'
data_folder = 'Nano-data/datasets'
test_folder = 'Nano-data/testsets'

# Create the 'datasets' folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    print(f"Created 'datasets' folder at '{data_folder}'")

if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    print(f"Created 'testsets' folder at '{test_folder}'")


# Get a list of all CSV files in the input folder
file_list = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Iterate through each file
for file_name in file_list:
    # Read the CSV file
    df = pd.read_csv(os.path.join(input_folder, file_name), sep=';')

    # Make changes to the dataframe (for example, add a new column)
    print(df)

    # Save the modified dataframe to a new CSV file in the output folder
    if "test" in file_name:
        output_file_name = os.path.join(test_folder, file_name)    
    else: 
        output_file_name = os.path.join(data_folder, file_name)

    df.to_csv(output_file_name, index=False, sep=',')

    print(f"File '{file_name}' processed and saved to '{output_file_name}'")

### First, split the dataset (Nano-data) into two datasets: datasets for training and testsets for testing.
    
    ### Now, Make trainset from csv files of datasets and testset from csv files of testsets.