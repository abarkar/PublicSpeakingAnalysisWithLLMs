import os
import pandas as pd
import shutil

# Function to clean up the .txt filenames
def clean_filename(filename):
    filename = filename.replace('output', '')  # Remove 'output' prefix
    filename = filename.replace(',', '')       # Remove commas
    filename = filename.replace(' ', '')       # Remove spaces
    filename = filename.replace('-', '')       # Remove spaces
    filename = filename.replace('transcript', '')       # Remove spaces
    return filename

# Load the list_ID.csv which contains the mapping of old file names to new IDs
list_id_df = pd.read_csv('./list_ID.csv', sep=',')  # Adjust the path if necessary
# print(list_id_df.head)
list_id_df['transcript ID'] = list_id_df['transcript ID'].apply(clean_filename)
# print(list_id_df.head)

# Folder containing the original transcript data
source_folder = './transcripts'

# Folder to save the renamed files
destination_folder = './transcript_renamed'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)


# Walk through the transcripts folder to find all .txt files
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.txt'):
            # Extract city and clip information from the folder structure
            city = os.path.basename(os.path.dirname(root))
            clip = os.path.basename(root)
            # print(city)
            # print(clip)
            
            # Clean up the current file name
            cleaned_filename = clean_filename(file.replace('.txt', ''))

            # Try to find the matching row in the list_ID.csv
            matching_row = list_id_df[
                (list_id_df['city'] == city) &
                (list_id_df['category'] == clip) &
                (list_id_df['transcript ID'].str.replace('output', '').replace('transcript', '').replace(',', '').replace(' ', '').replace('-', '') == cleaned_filename)
            ]

            if not matching_row.empty:
                new_id = matching_row['ID'].values[0]  # Get the new ID for renaming

                # Construct the destination folder structure (clip-based)
                new_clip_folder = os.path.join(destination_folder, clip)
                if not os.path.exists(new_clip_folder):
                    os.makedirs(new_clip_folder)

                # Construct the source and destination file paths
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(new_clip_folder, f'{new_id}.txt')

                # Copy and rename the file
                shutil.copy(old_file_path, new_file_path)
                # print(f'Renamed and moved: {old_file_path} -> {new_file_path}')
            else:
                print(f'No matching row found for: {file} under the name  {cleaned_filename} with {city} and {clip}')
