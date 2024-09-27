import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Load LLM features
llm_features_path = './LLM_features/LLM_features.csv'
llm_features = pd.read_csv(llm_features_path)

# Load lexical features
lexical_features_path = './lexical_features/lexical_features.csv'
lexical_features = pd.read_csv(lexical_features_path)

# Load ground truth scores and rename 'presentation' column to 'ID'
gt_scores_path = './MT_Labels_rms.csv'
ground_truth = pd.read_csv(gt_scores_path, sep=";")
ground_truth.rename(columns={'presentation': 'ID'}, inplace=True)
ground_truth=ground_truth[["persuasiveness_rms", "ID"]]
ground_truth["persuasiveness_rms"] = ground_truth["persuasiveness_rms"].str.replace(',', '.').astype(float)
print(ground_truth.columns)

# Merge data frames on 'ID' using inner join to only keep rows with common IDs across all datasets
merged_data = pd.merge(pd.merge(llm_features, lexical_features, on='ID', how='inner'), ground_truth, on='ID', how='inner')
merged_data =merged_data.drop('ID', axis=1)

# Compute Spearman and Pearson correlations
spearman_corr = merged_data.corr(method='spearman')
pearson_corr = merged_data.corr(method='pearson')


# Plotting the Spearman correlation matrix
plt.figure(figsize=(24, 24))  # Increased from 20x20 to 24x24 for larger cells

# Create the heatmap with annotations
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
            annot_kws={'size': 12})  # Adjust annot_kws size as needed

# Setting font sizes
plt.title('Spearman Correlation Matrix', fontsize=20)  # Title font size
plt.xticks(rotation=90, ha='right', fontsize=20)  # X-axis tick label font size
plt.yticks(rotation=0, fontsize=20)  # Y-axis tick label font size



# Extracting specific correlations
# Correlations of all features with ground truth data 'persuasiveness_rms'
spearman_gt_corr = spearman_corr['persuasiveness_rms'].drop('persuasiveness_rms')
pearson_gt_corr = pearson_corr['persuasiveness_rms'].drop('persuasiveness_rms')

# Display the correlation results
# print("Spearman Correlation with Ground Truth Persuasiveness:")
# print(spearman_gt_corr)
# print("\nPearson Correlation with Ground Truth Persuasiveness:")
# print(pearson_gt_corr)

# Save correlation results to a CSV file (optional)
spearman_gt_corr.to_csv('./correlation_results/spearman_correlation.csv')
pearson_gt_corr.to_csv('./correlation_results/pearson_correlation.csv')


# Performing ANOVA for each LLM feature against 'persuasiveness_rms'
anova_results = {}
for column in llm_features.columns:
    if column != 'ID':  # Assuming 'ID' is not a feature to test
        print("criteria:", column)
        # Prepare data for ANOVA
        group_data = merged_data.groupby(column)['persuasiveness_rms'].apply(list)
        # Perform One-Way ANOVA

        anova_result = f_oneway(*group_data)
        anova_results[column] = anova_result

        # Creating a boxplot for each feature
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=column, y='persuasiveness_rms', data=merged_data)
        # Setting font sizes
        plt.title(f"Persuasiveness Scores by {column}\nANOVA F={anova_result.statistic:.2f}, p={anova_result.pvalue:.3f}", fontsize=20)  # Set title font size
        plt.xlabel(f'{column} Category', fontsize=20)  # Set x-axis label font size
        plt.ylabel('Persuasiveness Scores', fontsize=20)  # Set y-axis label font size

        # Setting tick label sizes
        plt.xticks(rotation=45, fontsize=20)  # Set x-axis tick label font size
        plt.yticks(fontsize=20)  # Set y-axis tick label font size

        plt.tight_layout()
        plt.savefig(f'./correlation_results/persuasiveness_scores_by_{column}.png')
        plt.show()

# Save ANOVA results to a DataFrame and then to a CSV file
anova_df = pd.DataFrame([(k, v.statistic, v.pvalue) for k, v in anova_results.items()], columns=['Feature', 'F-Statistic', 'P-Value'])
anova_df.to_csv('./correlation_results/anova_results.csv', index=False)