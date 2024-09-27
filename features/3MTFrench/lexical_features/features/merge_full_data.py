import pandas as pd
import os

def concatenate_with_ratings(folder_path):
    # Load all_features.csv file
    all_features_path = os.path.join(folder_path, 'all_features.csv')
    all_features_df = pd.read_csv(all_features_path)

    # Load MT_aggregated_ratings.csv file
    ratings_path = os.path.join(folder_path, 'MT_aggregated_ratings.csv')
    ratings_df = pd.read_csv(ratings_path)

    # Filter rows in MT_aggregated_ratings.csv where 'clip' column equals 'full'
    filtered_ratings_df = ratings_df[ratings_df['clip'] == 'full']

    # Merge the filtered ratings with all_features on 'ID' column
    merged_df = pd.merge(all_features_df, filtered_ratings_df, on='ID', how='inner')

    # Save the final merged dataframe to a new CSV file
    output_path = os.path.join(folder_path, 'all_features_with_ratings.csv')
    merged_df.to_csv(output_path, index=False)

    print(f"The files have been concatenated and saved to {output_path}")

# Specify the folder where the CSV files are located
folder_path = './'
concatenate_with_ratings(folder_path)
