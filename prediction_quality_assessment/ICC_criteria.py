import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from pingouin import intraclass_corr
import seaborn as sns
import matplotlib.pyplot as plt

criteria_groups = ["conciseness", "dimention evaluation", "form evaluation", "language level", "negative language", "storytelling"]
groups= ["conciseness", "dimentions", "form", "language_level", "negative_language", "storytelling"]
scores={"conciseness": ["length", "redundancy"], "dimentions":["persuasiveness", "clarity", "authenticity"], "form":["topic", "structure"], "language_level":["languageLVL", 'passiveV'], "negative_language":["negLang"] ,  "storytelling":["metaphor", "discours"]}
test_numbers = ['1', '2', '3']
models = ['llama2', 'mistral']
csv_folder = './criteria_evaluation/results/results/'

icc_results = {}
cur = 1
# Create an empty list to store merged dataframes
all_merged_dfs = []

for group, criteria_group in zip(groups, criteria_groups):
    print("group", group)
    print("criteria_group", criteria_group)

    merged_df = pd.DataFrame()
    for test_number in test_numbers:
        for model in models:
            csv_filename = f'{group}_{test_number}_LLM_answers_{model}_7b.csv'
            path = os.path.join(csv_folder, criteria_group, csv_filename)
            if os.path.exists(path):
                predicted_df = pd.read_csv(path)
                predicted_df['rater'] = cur
                predicted_df['criteria'] = criteria_group  # Add criteria group column
                merged_df = pd.concat([merged_df, predicted_df], ignore_index=True)
                cur += 1
    cur = 1

    # Append merged_df to the list of all merged dataframes
    all_merged_dfs.append(merged_df)

    # Calculate ICC for each score in the criteria group
    for score_name in scores[group]:
        # merged_df[score_name] = pd.to_numeric(merged_df[score_name].str.replace(',', '.'), errors='coerce')
        icc_result = intraclass_corr(data=merged_df, targets='transcript ID', raters='rater', ratings=score_name, nan_policy='omit')
        icc_results[(criteria_group, score_name)] = icc_result

# Concatenate all merged_df dataframes
        
all_merged_df = pd.concat(all_merged_dfs)

all_merged_df = all_merged_df[['transcript ID', 'model', 'length',
       'redundancy', 'rater', 'criteria', 'persuasiveness', 'clarity',
       'authentisity', 'authenticity', 'topic', 'structure', 'languageLVL',
       'passiveV', 'negLang', 'metaphor', 'discours']]
# print(all_merged_df.columns)

# [["length", "redundancy","persuasiveness", "clarity", "authenticity","topic", "structure","languageLVL", 'passiveV',"negLang","metaphor", "discours"]]]


# Save ICC results to CSV file with semicolon as separator
output_icc_filename = 'ICC_results.csv'
output_icc_path = os.path.join(csv_folder, output_icc_filename)
with open(output_icc_path, 'w') as f:
    f.write("Criteria Group;Score;ICC(3,1);ICC(3,k)\n")
    for (criteria_group, score), icc_result in icc_results.items():
        f.write(f"{criteria_group};{score};{icc_result['ICC'].values[0]:.4f};{icc_result['ICC'].values[1]:.4f}\n")

print("ICC results saved to", output_icc_path)

# Concatenate all merged_df dataframes for each criteria group
# all_merged_df = pd.concat([merged_df for criteria_group, merged_df in merged_dfs.items()])

# Select only numeric columns
numeric_columns = all_merged_df.select_dtypes(include=np.number)

# Calculate correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Scores from Different Criteria Groups')
plt.xlabel('Criteria Groups')
plt.ylabel('Criteria Groups')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Save correlation matrix plot to a PNG file
correlation_matrix_filename = 'Correlation_Matrix.png'
correlation_matrix_path = os.path.join(csv_folder, correlation_matrix_filename)
plt.savefig(correlation_matrix_path)

print("Correlation matrix plot saved to", correlation_matrix_path)
