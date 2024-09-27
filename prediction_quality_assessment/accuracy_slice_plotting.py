import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


model = "llama2"
dimension = "persuasiveness"
version = '4'
test_number = '1'
model_name_with_test_number = f"{model}-{test_number}"
csv_folder = os.path.join('./dimension_evaluation/results', dimension, version, model_name_with_test_number)
csv_filename = f'dimentions_all_files_{dimension}_evaluation_prompt_{version}_test_{test_number}_LLM_answers_{model}_7b.csv'


# csv_filename = f'dimentions_all_files_evaluation_{test_number}_LLM_answers_{model}_7b.csv'
path = os.path.join(csv_folder, csv_filename)

# Load predicted scores
predicted_df = pd.read_csv(path, sep=',')
print(predicted_df.head())

# Load ground-truth scores
ground_truth_df = pd.read_csv("./MT_Labels_rms.csv", sep=';')
print(ground_truth_df.head())


predicted_df["ID"]=predicted_df["ID"].str.lower()
ground_truth_df["presentation"]=ground_truth_df["presentation"].str.lower()
# # Calculate epsilon
# epsilon = (ground_truth_df[dimension + '_rms'].max() - ground_truth_df[dimension + '_rms'].min()) / 100


# Merge the two dataframes based on performance ID and category
merged_df = pd.merge(predicted_df, ground_truth_df, left_on=["ID", "category"], right_on=["presentation", "video"])



# # Calculate epsilon
# epsilon = (merged_df[dimension + '_rms'].max() - merged_df[dimension + '_rms'].min()) / 100

# Define a function to calculate metrics for each slice
def calculate_metrics(slice_df):
    # Convert 'score' and 'global_rms' columns to numeric
    # slice_df['score'] = pd.to_numeric(slice_df['score'], errors='coerce')
    # slice_df[dimension + '_rms'] = pd.to_numeric(slice_df[dimension + '_rms'].str.replace(',', '.'), errors='coerce')
    # # Drop rows with NaN values resulting from conversion
    # slice_df.dropna(subset=['score', dimension + '_rms'], inplace=True)

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
forbiden_ID=['als10', 'cfe10', 'uca01', 'uca02', 'uca03', 'uca04', 'uca05', 'uca06', 'uca07', 'uca08', 'uca09', 'uca10', 'uca11', 'uca12', 'uca13', 'uca14', 'uca15']

accepted_ID = pd.read_csv("/home/alisa/Documents/Mock-Up_sample/Mock-Up_sample/REVITALISE/results/MT/regression/persuasiveness/full/linear/persuasiveness_full_linear_predictions.csv")["ID"].str.lower().unique()
print("accepted_ID: ", accepted_ID)
# Iterate over slices
for slice_name, slice_df in merged_df.groupby('category'):
    if slice_name == "full" :
        print(slice_df.shape)
        print(np.dtype(slice_df['quality of answer']))
        print(slice_df[~slice_df['ID'].isin(forbiden_ID)])

        slice_df = slice_df[(slice_df['quality of answer']==1) & ~slice_df['ID'].isin(forbiden_ID) ]
        slice_df = slice_df[slice_df["ID"].str.lower().isin(accepted_ID)]
        
        slice_df['score'] = pd.to_numeric(slice_df['score'], errors='coerce')
        slice_df[dimension + '_rms'] = pd.to_numeric(slice_df[dimension + '_rms'].str.replace(',', '.'), errors='coerce')
        # Drop rows with NaN values resulting from conversion
        slice_df.dropna(subset=['score', dimension + '_rms'], inplace=True)

        # Calculate epsilon
        epsilon = (slice_df[dimension + '_rms'].max() - slice_df[dimension + '_rms'].min()) / 100
#  train_texts

        metrics = calculate_metrics(slice_df)
        for metric_name, metric_value in metrics.items():
            results.append({
                "Slice": slice_name,
                "Metric": metric_name,
                "Value": metric_value
            })
        # Calculate density of ground truth scores
        slice_df['density'] = slice_df[dimension + '_rms'].apply(lambda x: len(slice_df[(slice_df[dimension + '_rms'] >= x - epsilon) & (slice_df[dimension + '_rms'] <= x + epsilon)]))
        
        # Normalize density to [0, 1]
        slice_df['density'] = slice_df['density'] / slice_df['density'].max()
        
        # Plot ground truth vs predicted scores with color reflecting density
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=slice_df[dimension + '_rms'], y=slice_df['score'], alpha=0.5, hue=slice_df['density'], palette='viridis')
        
        # Setting font sizes
        plt.title(f"Predicted vs Ground Truth for {model}", fontsize=20)  # Set title font size
        plt.xlabel("Ground Truth", fontsize=20)  # Set x-axis label font size
        plt.ylabel("Predicted", fontsize=20)  # Set y-axis label font size

        # Setting tick label sizes
        plt.xticks(fontsize=20)  # Set x-axis tick label font size
        plt.yticks(fontsize=20)  # Set y-axis tick label font size

        
        plt.grid(True)
        plt.xlim(0, 6)  # Limit x-axis to the range of ground truth scores (0 to 6)
        plt.ylim(0, 6)  # Limit y-axis to the range of predicted scores (0 to 6)
        plt.plot([0, 6], [0, 6], color='red', linestyle='--')  # Add a diagonal line for reference
        # Save the plot as PNG file
        plot_filename = f"plot_{dimension}_{version}test_set{test_number}_{slice_name}.png"
        plt.savefig(os.path.join(csv_folder, plot_filename))
        plt.close()  # Close the plot to free up memory

# Create DataFrame for results
results_df = pd.DataFrame(results)

# Save DataFrame to CSV file
output_csv_filename = "all_metrics_" + dimension + "_" + version + "_test_set_" + test_number + ".csv"
results_df.to_csv(os.path.join(csv_folder, output_csv_filename), index=False)

print("Metrics saved to", output_csv_filename)
