import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from pingouin import intraclass_corr

models = ["gpt-4o-mini"]
dimension = "persuasiveness"
version = '1'
test_numbers = ['1', '2','3']
csv_folder = os.path.join('./dimension_evaluation/results', dimension, version)
reg_models_list=['ridge', 'linear', 'random_forest', 'ElasticNet', 'lasso']



# Function to calculate ICC for specific pairs of raters
def calculate_pairwise_icc(dataframe, raters_list, dimension, csv_folder, suffix):
    # Filter the dataframe for the specified raters
    filtered_df = dataframe[dataframe['rater'].isin(raters_list)]
    
    # Calculate ICC
    icc_result = intraclass_corr(data=filtered_df, targets='ID', raters='rater', ratings='score', nan_policy='omit').round(4)
    
    # Save ICC results to CSV file
    raters_str = '_'.join(map(str, raters_list))
    output_icc_filename = f'ICC_pairwise_{dimension}_{raters_str}_{suffix}.csv'
    output_icc_path = os.path.join(csv_folder, output_icc_filename)
    icc_result.to_csv(output_icc_path, index=False)
    
    print(f"ICC results for raters {raters_str} saved to", output_icc_path)



forbiden_ID=['als10', 'cfe10', 'uca01', 'uca02', 'uca03', 'uca04', 'uca05', 'uca06', 'uca07', 'uca08', 'uca09', 'uca10', 'uca11', 'uca12', 'uca13', 'uca14', 'uca15']
accepted_ID = pd.read_csv("/home/alisa/Documents/Mock-Up_sample/Mock-Up_sample/REVITALISE/results/MT/regression/persuasiveness/full/linear/persuasiveness_full_linear_predictions.csv")["ID"].str.lower().unique()

# Load ground-truth scores
ground_truth_df = pd.read_csv("./MT_aggregated_ratings.csv", sep=',')
ground_truth_df = ground_truth_df[(ground_truth_df['aggregationMethod']=="rms") & (ground_truth_df["clip"]=="full")]

# Create an array of pairs
model_with_test_number = [(x, y) for x in models for y in test_numbers]
rater_count=0
# Load predicted scores for each test number and append them
merged_df = pd.DataFrame()
for pair in model_with_test_number:
    rater_count+=1
    csv_filename = f'COMMERTIAL_dimentions_all_files_{dimension}_evaluation_prompt_{version}_test_{pair[1]}_LLM_answers_{pair[0]}.csv'
    path = os.path.join(csv_folder, f"{pair[0]}-{pair[1]}", csv_filename)
    predicted_df = pd.read_csv(path, sep=',')
    # predicted_df = predicted_df[(predicted_df['category']=="full") &(predicted_df['ID'].str.lower().isin(accepted_ID))&(~predicted_df['ID'].str.lower().isin(forbiden_ID))]
    predicted_df = predicted_df[(predicted_df['clip']=="full")&(~predicted_df['ID'].str.lower().isin(forbiden_ID))]
    print("Considered data: ", np.shape(predicted_df))
    print("rater: ", rater_count)
    print("pair: ", pair)
    predicted_df['ID']=predicted_df['ID'].str.lower()
    predicted_df['rater'] = rater_count  # Assign rater value
    merged_df = pd.concat([merged_df, predicted_df], ignore_index=True)
    # print(np.shape(merged_df))

merged_df = merged_df[['ID', 'clip', 'rater', 'score']]

# reg_pred_data = []
# for rgs in reg_models_list:
#     rater_count+=1
#     reg_predictions = pd.read_csv(f"/home/alisa/Documents/Mock-Up_sample/Mock-Up_sample/REVITALISE/results/MT/regression/persuasiveness/full/{rgs}/persuasiveness_full_{rgs}_predictions.csv")
#     reg_predictions['clip']="full"
#     reg_predictions['rater'] = rater_count  # Assign rater value
#     reg_predictions['ID']=reg_predictions['ID'].str.lower()
#     merged_df = pd.concat([merged_df, reg_predictions], ignore_index=True)
#     # print(np.shape(merged_df))


rater_count+=1
# print(np.shape(merged_df))
# merged_df = merged_df[['ID','rater', 'score']]

# print(np.shape(merged_df))

# # Save merged and appended dataframe
# output_merged_filename = f'merged_appended_{dimension}_{version}.csv'
# output_merged_path = os.path.join(csv_folder, output_merged_filename)
# merged_df.to_csv(output_merged_path, index=False)

# Construct DataFrame for ground truth data
ground_truth_data = []
for index, row in ground_truth_df.iterrows():
    ground_truth_data.append({
        'ID': str(row['ID']).lower(),
        'clip': row['clip'],
        'score': row[dimension],
        'rater': rater_count  # Assign rater value for ground truth
    })
ground_truth_df_full = pd.DataFrame(ground_truth_data)
ground_truth_df_full=ground_truth_df_full[ground_truth_df_full['clip']=="full"]
# ground_truth_df_full=ground_truth_df_full[(ground_truth_df_full['ID'].str.lower().isin(accepted_ID))&(~ground_truth_df_full['ID'].str.lower().isin(forbiden_ID))]
ground_truth_df_full=ground_truth_df_full[(~ground_truth_df_full['ID'].str.lower().isin(forbiden_ID))]
# ground_truth_df_full['score'] = pd.to_numeric(ground_truth_df_full['score'].str.replace(',', '.'), errors='coerce')

# Concatenate ground truth data to merged_df
merged_df = pd.concat([merged_df, ground_truth_df_full], ignore_index=True)

# print(np.shape(merged_df))

# print(len(merged_df[merged_df['rater']==7]))

# merged_df['score'] = pd.to_numeric(merged_df['score'].str.replace(',', '.'), errors='coerce')
# Drop rows with NaN values resulting from conversion
merged_df.dropna(subset=['score'], inplace=True)


# Save merged and appended dataframe
output_merged_filename = f'COMMERTIAL_full_data_merged_appended_{dimension}_{version}.csv'
output_merged_path = os.path.join(csv_folder, output_merged_filename)
merged_df.to_csv(output_merged_path, index=False)

print("Merged and appended dataframe saved to", output_merged_path)


# Save ICC results to CSV file
output_icc_filename = f'COMMERTIAL_ICC_{dimension}.csv'
output_icc_path = os.path.join(csv_folder, output_icc_filename)
# Calculate ICC for each slice
slices = merged_df.groupby('clip')
# icc_results = {}
for slice_name, slice_df in slices:
    # print(slice_df.head())
    icc_result = intraclass_corr(data=slice_df, targets='ID', raters='rater', ratings='score', nan_policy='omit').round(4)
    icc_result.to_csv(output_icc_path, index=False)
    
    # icc_results[slice_name] = icc_result


# with open(output_icc_path, 'w') as f:
#     for slice_name, icc_result in icc_results.items():
#         f.write(f"Slice {slice_name} ICC(3,1): {icc_result['ICC'].values[0]:.4f}\n")
#         f.write(f"Slice {slice_name} ICC(3,k): {icc_result['ICC'].values[1]:.4f}\n")

print("ICC results saved to", output_icc_path)


# Llama2 models only
# calculate_pairwise_icc(merged_df, [1, 2, 3], dimension, csv_folder, "only_llama2")  # Adjust the rater numbers accordingly

# # Mistral models only
# calculate_pairwise_icc(merged_df, [4, 5, 6], dimension, csv_folder, "only_mistral")  # Adjust the rater numbers accordingly

# llama3 models only
calculate_pairwise_icc(merged_df, [1, 2, 3], dimension, csv_folder, "only_COMMERTIAL")  # Adjust the rater numbers accordingly


# # Llama2 models and ground truth ratings
# calculate_pairwise_icc(merged_df, [1, 2, 3, rater_count], dimension, csv_folder, "llama2VSGT")  # 'rater_count' is the ground truth

# # Mistral models and ground truth ratings
# calculate_pairwise_icc(merged_df, [4, 5, 6, rater_count], dimension, csv_folder, "mistralVSGT")

# # Mistral and Llama2 models only (excluding ground truth)
# calculate_pairwise_icc(merged_df, [1, 2, 3, 4, 5, 6], dimension, csv_folder, "llama2VSmistral")

# # Mistral and Llama2 and regression models only (excluding ground truth)
# calculate_pairwise_icc(merged_df, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dimension, csv_folder, "llama2VSmistralVSReg")

# # regression models only (excluding ground truth)
# calculate_pairwise_icc(merged_df, [7, 8, 9, 10, 11], dimension, csv_folder, "only_regression")

# # regression models and ground truth ratings
# calculate_pairwise_icc(merged_df, [7, 8, 9, 10, 11, rater_count], dimension, csv_folder, "RegVSGT")

# # RG models and ground truth ratings
# calculate_pairwise_icc(merged_df, [7, rater_count], dimension, csv_folder, "RGVSGT")
# #  models and ground truth ratings
# calculate_pairwise_icc(merged_df, [8, rater_count], dimension, csv_folder, "LRVSGT")
# # RFR models and ground truth ratings
# calculate_pairwise_icc(merged_df, [9, rater_count], dimension, csv_folder, "RFRVSGT")
# # RFR models and ground truth ratings
# calculate_pairwise_icc(merged_df, [10, rater_count], dimension, csv_folder, "eNetVSGT")
# # RFR models and ground truth ratings
# calculate_pairwise_icc(merged_df, [11, rater_count], dimension, csv_folder, "LSVSGT")

# RFR models and ground truth ratings
# calculate_pairwise_icc(merged_df, [4,  9], dimension, csv_folder, "RFRVSmistral")


# reg_models_list=['ridge', 'linear', 'random_forest', 'ElasticNet', 'lasso']
