
#This script can be run to install the NIH Dataset into the scratch dir. In my experience
#it takes about 30-40 min for this task

#make sure to add your kaggle api keys to .kaggle dir and also install kagglehub to the env


import kagglehub
import os

# Define the handle for the dataset you want
dataset_handle = 'nih-chest-xrays/data'

# Define where you want to save the data on the NSCC
# Using os.path.expanduser('~') is a robust way to get your home directory path
download_path = os.path.expanduser('~/scratch/nih_chest_xray_dataset')

print(f"--- Starting Kaggle dataset download ---")
print(f"Dataset: {dataset_handle}")
print(f"Destination: {download_path}")

# This is the command that downloads the data
# It will create the destination folder for you
# It also automatically unzips the data
path = kagglehub.dataset_download(
    dataset_handle,
    path=download_path
)

print(f"--- Download and extraction complete ---")
print(f"Dataset is located at: {path}")
