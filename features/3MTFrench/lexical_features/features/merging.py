import os
import pandas as pd

def merge_csv_files(folder_path):
    # List to hold dataframes
    dataframes = []

    # Loop through all the CSV files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Merge all dataframes on 'ID' column, using outer join to keep all data
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on='ID', how='outer')

    # Save the final merged dataframe to a CSV file
    output_path = os.path.join(folder_path, 'all_features.csv')
    merged_df.to_csv(output_path, index=False)

    print(f"All CSV files have been merged and saved to {output_path}")

# Specify the folder where the CSV files are located
folder_path = './'
merge_csv_files(folder_path)
