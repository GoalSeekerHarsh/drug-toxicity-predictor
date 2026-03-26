"""
simple_load.py - A beginner-friendly script to load and inspect the Tox21 dataset.
"""

# Import pandas, the standard Python library for data manipulation
import pandas as pd

# Define the path to our dataset
file_path = "../data/raw/tox21.csv"

# Load the dataset from the CSV file into a pandas 'DataFrame' (like a spreadsheet)
print("Loading data...")
df = pd.read_csv(file_path)

# Print the number of rows (compounds) and columns (features/labels)
print(f"\n1. Number of rows: {df.shape[0]}")
print(f"2. Number of columns: {df.shape[1]}")

# Print the names of all the columns in the dataset
print(f"\n3. Column names:\n{df.columns.tolist()}")

# Print the first 5 rows of the dataset to see what the data actually looks like
print("\n4. First 5 rows:")
print(df.head(5))

# Calculate and print the total number of missing values across the entire dataset
print(f"\n5. Total missing values: {df.isna().sum().sum()}")
