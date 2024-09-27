import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

model = "llama2"
dimension = "persuasiveness"
version = '1'
test_number = '2'
csv_folder = os.path.join('./dimension_evaluation/results', dimension, version, test_number)
csv_filename = f'dimentions_all_files_{dimension}_evaluation_prompt_{version}_test_{test_number}_LLM_answers_{model}_7b.csv'


# csv_filename = f'dimentions_all_files_evaluation_{test_number}_LLM_answers_{model}_7b.csv'
path = os.path.join(csv_folder, csv_filename)

# Load predicted scores
predicted_df = pd.read_csv(path, sep=',')
print(predicted_df.head())

# Load ground-truth scores
ground_truth_df = pd.read_csv("./MT_Labels_rms.csv", sep=';')
print(ground_truth_df.head())


# Merge the two dataframes based on performance ID and category
merged_df = pd.merge(predicted_df, ground_truth_df, left_on=["ID", "category"], right_on=["presentation", "video"])

# Define a function to calculate metrics for each slice
def calculate_metrics(slice_df):
    # Convert 'score' and 'global_rms' columns to numeric
    slice_df['score'] = pd.to_numeric(slice_df['score'], errors='coerce')
    slice_df[dimension + '_rms'] = pd.to_numeric(slice_df[dimension + '_rms'].str.replace(',', '.'), errors='coerce')
    # Drop rows with NaN values resulting from conversion
    slice_df.dropna(subset=['score', dimension + '_rms'], inplace=True)

    # Calculate metrics
    mae = mean_absolute_error(slice_df[dimension + '_rms'], slice_df['score'])
    mse = mean_squared_error(slice_df[dimension + '_rms'], slice_df['score'])
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((slice_df[dimension + '_rms'] - slice_df['score']) / slice_df[dimension + '_rms'])) * 100
    r2 = r2_score(slice_df[dimension + '_rms'], slice_df['score'])
    n = len(slice_df)
    p = len(slice_df.columns) - 1
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    mpe = np.mean((slice_df[dimension + '_rms'] - slice_df['score']) / slice_df[dimension + '_rms']) * 100
    medae = median_absolute_error(slice_df[dimension + '_rms'], slice_df['score'])
    theils_u = theil_u_statistic(slice_df[dimension + '_rms'], slice_df['score'])
    cochran_q = cochran_q_statistic(slice_df[dimension + '_rms'], slice_df['score'])
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
        "MPE": mpe,
        "MedAE": medae,
        "Theils_U": theils_u,
        "Cochran_Q": cochran_q
    }

# Define Theil's U statistic function
def theil_u_statistic(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denominator = np.sqrt(np.mean(y_true ** 2))
    return numerator / denominator

# Define Cochran's Q statistic function
def cochran_q_statistic(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return numerator / denominator

# List to store results for each slice
results = []

# Iterate over slices
for slice_name, slice_df in merged_df.groupby('category'):
    metrics = calculate_metrics(slice_df)
    for metric_name, metric_value in metrics.items():
        results.append({
            "Slice": slice_name,
            "Metric": metric_name,
            "Value": metric_value
        })

# Create DataFrame for results
results_df = pd.DataFrame(results)

# Save DataFrame to CSV file
output_csv_filename = "all_metrics_" + dimension + "_" + version + "_" + test_number + ".csv"
results_df.to_csv(os.path.join(csv_folder, output_csv_filename), index=False)

print("Metrics saved to", output_csv_filename)
