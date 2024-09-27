import os
import pandas as pd
import numpy as np

# Dictionary of criteria groups and their corresponding criteria
criteria_dict = {
    "conciseness": ["length", "redundancy"],
    "form": ["topic", "structure"],
    "language_level": ["languageLVL", "passiveV"],
    "negative_language": ["negLang"],
    "storytelling": ["metaphor", "discours"]
}

# Variables you can define
version = '1'
model = 'mistral'
test_number = '1'

def read_feature_file(criteria_group, criteria, version, model, test_number):
    # Construct the file path
    base_path = f'./results/{criteria_group}/{criteria}/{version}/{model}-{test_number}'
    filename = f'dimentions_all_files_{criteria_group}_{criteria}_evaluation_prompt_{version}_test_{test_number}_LLM_answers_{model}_7b.csv'
    file_path = os.path.join(base_path, filename)
    
    # Read the CSV file
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Set 'score' to NaN where 'quality of answer' is 0
        df['score'] = df.apply(lambda row: np.nan if row['quality of answer'] == 0 else row['score'], axis=1)
        
        # Rename 'score' to a unique name to avoid column name clashes later
        new_col_name = f'{criteria}'
        df.rename(columns={'score': new_col_name}, inplace=True)
        
        # Return only the 'ID' and the new score column
        return df[['ID', new_col_name]]
    else:
        return pd.DataFrame()

# Prepare to concatenate all features
all_features = []

# Iterate over the criteria dictionary to access all possible files
for criteria_group, criteria_list in criteria_dict.items():
    for criteria in criteria_list:
        feature_df = read_feature_file(criteria_group, criteria, version, model, test_number)
        if not feature_df.empty:
            all_features.append(feature_df)

# Merge all feature DataFrames on 'ID' if there are any DataFrames to merge
if all_features:
    from functools import reduce
    final_df = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), all_features)
    
    # Ensure the output directory exists
    output_dir = './LLM_features'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the concatenated DataFrame
    final_df.to_csv(os.path.join(output_dir, 'LLM_features.csv'), index=False)
else:
    print("No feature files were found or all were empty.")
