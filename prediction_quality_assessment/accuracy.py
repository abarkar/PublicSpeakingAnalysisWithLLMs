import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

model = "llama2"
dimention = "global"
version = '1'
test_number = '1'
csv_folder = os.path.join('./dimension_evaluation/results', dimention, version, test_number)
# csv_filename = f'dimentions_all_files_{dimention}_evaluation_prompt_{version}_test_{test_number}_LLM_answers_{model}_7b.csv'
csv_filename = f'dimentions_all_files_evaluation_{version}_LLM_answers_{model}_7b_edited.csv'
path = os.path.join(csv_folder, csv_filename)
print(path)
# model_name_with_test_number = f"{json_subfolder}-{test_number}"
dimention_rms = dimention+"_rms"
# Load predicted scores
predicted_df = pd.read_csv(path, sep=';')
print(predicted_df.head())
# Load ground-truth scores

ground_truth_df = pd.read_csv("./MT_Labels_rms.csv", sep=';')
print(ground_truth_df.head())

print(np.shape(predicted_df))
print(np.shape(ground_truth_df))


# Merge the two dataframes based on performance ID and category
merged_df = pd.merge(predicted_df, ground_truth_df, left_on=["ID", "category"], right_on=["presentation", "video"])
print(np.shape(merged_df))
print(merged_df['score'])


# Save merged_df to CSV file
merged_df.to_csv(os.path.join(csv_folder, "./merged.csv"), index=False)


# Convert 'score' and 'global_rms' columns to numeric
merged_df['score'] = pd.to_numeric(merged_df['score'], errors='coerce')
print(np.shape(merged_df))
merged_df[dimention_rms] = pd.to_numeric(merged_df[dimention_rms].str.replace(',', '.'), errors='coerce')
print(np.shape(merged_df))
# Drop rows with NaN values resulting from conversion
merged_df.dropna(subset=['score', dimention_rms], inplace=True)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(merged_df[dimention_rms], merged_df['score'])

# Mean Squared Error (MSE)
mse = mean_squared_error(merged_df[dimention_rms], merged_df['score'])

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(merged_df[dimention_rms], merged_df['score'])

# Coefficient of Determination (R²)
r2 = r2_score(merged_df[dimention_rms], merged_df['score'])

# Adjusted R²
n = len(merged_df)
p = len(merged_df.columns) - 1  # Number of predictors (excluding the intercept)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Mean Percentage Error (MPE)
mpe = np.mean((merged_df[dimention_rms] - merged_df['score']) / merged_df[dimention_rms]) * 100

# Median Absolute Error (MedAE)
medae = median_absolute_error(merged_df[dimention_rms], merged_df['score'])

# Theil's U statistic
def theil_u_statistic(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denominator = np.sqrt(np.mean(y_true ** 2))
    return numerator / denominator
theils_u = theil_u_statistic(merged_df[dimention_rms], merged_df['score'])

# Cochran's Q statistic
def cochran_q_statistic(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return numerator / denominator
cochran_q = cochran_q_statistic(merged_df[dimention_rms], merged_df['score'])


# Create a DataFrame to store metrics and their values
metrics_df = pd.DataFrame({
    "metrics_name": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", 
                     "Mean Absolute Percentage Error (MAPE)", "Coefficient of Determination (R²)", "Adjusted R²:", "Mean Percentage Error (MPE):",  "Median Absolute Error (MedAE)", "Theil's U statistic:", "Cochran's Q statistic:"],
    "metrics_value": [mae, mse, rmse, mape, r2, adjusted_r2, mpe, medae, theils_u, cochran_q]
})

# Save DataFrame to CSV file
output_csv_filename = "metrics_" + dimention + "_" + version + "_" + test_number + ".csv"

metrics_df.to_csv(os.path.join(csv_folder, output_csv_filename), index=False)

print("Metrics saved to ./metrics.csv_"+ dimention + "_" + version +".csv")
