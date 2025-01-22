import pandas as pd
import os

# Directory path
directory = r"C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Master_Base"

# Get all CSV files in the directory
file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('_base_stats.csv')]

# Read and combine all dataframes
dataframes = []
for file_path in file_paths:
    # Extract player name from file path
    player_name = file_path.split("\\")[-1].replace("_base_stats.csv", "")
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all dataframes
master_base2 = pd.concat(dataframes, ignore_index=True)

# Optional: Save the combined dataframe to a new CSV file
master_base2.to_csv(r"C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Master_Base\master_base_stats.csv", index=False)
