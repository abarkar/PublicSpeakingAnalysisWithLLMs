import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Load LLM features
llm_features_path = './LLM_features/LLM_features.csv'
llm_features = pd.read_csv(llm_features_path)
#llm_features = pd.read_csv(llm_features_path).fillna(0)

# Load lexical features
lexical_features_path = './lexical_features/lexical_features.csv'
lexical_features = pd.read_csv(lexical_features_path)
#lexical_features = pd.read_csv(lexical_features_path).fillna(0)

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

print(merged_data.head())
print(merged_data.columns)


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
        # print(*group_data)
        number_of_groups = len(group_data)
        # print(number_of_groups)
        if (number_of_groups > 1):

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
            #plt.show()

# Save ANOVA results to a DataFrame and then to a CSV file
anova_df = pd.DataFrame([(k, v.statistic, v.pvalue) for k, v in anova_results.items()], columns=['Feature', 'F-Statistic', 'P-Value'])
anova_df.to_csv('./correlation_results/anova_results.csv', index=False)



#---- Saving Spearman and Pearson Correlation Results for llm_features Against lexical_features ----
# Merge llm_features and lexical_features on 'ID', and drop 'ID' before computing correlations
# ll_lex_merged = pd.merge(llm_features, lexical_features, on='ID').drop(columns='ID')

# # Compute Spearman and Pearson correlations for llm_features and lexical_features
# ll_lex_spearman_corr = ll_lex_merged.corr(method='spearman').loc[llm_features.columns.drop('ID'), lexical_features.columns.drop('ID')]
# ll_lex_pearson_corr = ll_lex_merged.corr(method='pearson').loc[llm_features.columns.drop('ID'), lexical_features.columns.drop('ID')]

# # Save the correlation results to CSV files
# ll_lex_spearman_corr.to_csv('./correlation_results/LLM_against_lexical_Spearman_correlation.csv')
# ll_lex_pearson_corr.to_csv('./correlation_results/LLM_against_lexical_Pearson_correlation.csv')

# # Iterate over each LLM feature
# print(lexical_features.columns)

# # Assert that 'ID' is unique in both DataFrames
# assert llm_features.columns.is_unique, "LLM features columns are not unique"
# assert lexical_features.columns.is_unique, "Lexical features columns are not unique"
for llm_feature in llm_features.columns.drop("ID"):
    # Select the current LLM feature data merged with all lexical features
    ll_lex_merged = pd.merge(llm_features[[llm_feature, 'ID']], lexical_features, on='ID', how='inner')
    ll_lex_merged=ll_lex_merged.drop(columns="ID")

# merged_llm_pred = merged_llm_pred.drop(columns='ID')

    # ll_lex_merged.columns.drop("ID")

    # print(ll_lex_merged.columns)
    # Compute Spearman and Pearson correlations for the current LLM feature against all lexical features
    spearman_corr_feature = ll_lex_merged.corr(method='spearman').loc[[llm_feature], lexical_features.columns.drop('ID')].transpose()
    pearson_corr_feature = ll_lex_merged.corr(method='pearson').loc[[llm_feature], lexical_features.columns.drop('ID')].transpose()

    # Save the correlation results to separate CSV files for each LLM feature
    spearman_corr_feature.to_csv(f'./correlation_results/{llm_feature}_against_lexical_Spearman_correlation.csv', header=['Correlation'])
    pearson_corr_feature.to_csv(f'./correlation_results/{llm_feature}_against_lexical_Pearson_correlation.csv', header=['Correlation'])




# ---- Performing One-Way ANOVA Between llm_features and lexical_features ---- 
anova_results_ll_lex = {}
for llm_col in llm_features.columns.drop('ID'):
    for lex_col in lexical_features.columns.drop('ID'):
        grouping_wrt_llm = merged_data.groupby(llm_col)
        # Prepare data for ANOVA
        groups = [group[lex_col].values for name, group in grouping_wrt_llm]
        # Perform One-Way ANOVA
        number_of_groups = len(grouping_wrt_llm)
        # print(number_of_groups)
        if (number_of_groups > 1):
            anova_result = f_oneway(*groups)
            anova_results_ll_lex[f'{llm_col}_vs_{lex_col}'] = anova_result

# Save ANOVA results to a DataFrame and then to a CSV file
anova_ll_lex_df = pd.DataFrame([(k, v.statistic, v.pvalue) for k, v in anova_results_ll_lex.items()], columns=['Feature Pair', 'F-Statistic', 'P-Value'])
anova_ll_lex_df.to_csv('./correlation_results/LLM_against_lexical_ANOVA.csv', index=False)


#---- Calculating Variation and Mean for Each Feature in llm_features and lexical_features ----
# Calculate variation and mean for llm_features and lexical_features
features_stats = pd.concat([llm_features.describe().loc[['mean', 'std'],:], lexical_features.describe().loc[['mean', 'std'],:]])
features_stats.to_csv('./correlation_results/features_variation_and_mean.csv')


#---- One-Way ANOVA for Predicted Values of Persuasiveness Against llm_features ----
# Load the predicted persuasiveness scores
predicted_scores_path = './results/persuasiveness/1/mistral-1/dimentions_all_files_persuasiveness_evaluation_prompt_1_test_1_LLM_answers_mistral_7b.csv'
predicted_scores = pd.read_csv(predicted_scores_path)
predicted_scores['score'] = predicted_scores['score'].astype(float)

# Merge with llm_features on 'ID'
merged_pred_scores = pd.merge(predicted_scores, llm_features, on='ID')

# Performing ANOVA for each LLM feature against predicted 'score'
anova_pred_results = {}
for column in llm_features.columns.drop('ID'):
    group_data = merged_pred_scores.groupby(column)['score'].apply(list)

    number_of_groups = len(group_data)
    print(number_of_groups)
    if (number_of_groups > 1):
        anova_result = f_oneway(*group_data)
        anova_pred_results[column] = anova_result

        # Creating a boxplot for each feature
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=column, y='score', data=merged_pred_scores)


        # Setting font sizes
        plt.title(f"Persuasiveness Scores by {column}\nANOVA F={anova_result.statistic:.2f}, p={anova_result.pvalue:.3f}", fontsize=20)
        plt.xlabel(f'{column} Category', fontsize=20)
        plt.ylabel('Persuasiveness Scores', fontsize=20)

        # Setting tick label sizes
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)

        plt.tight_layout()
        # Save the plot before showing it
        plt.savefig(f'./correlation_results/persuasiveness_scores_by_{column}.png')
        #plt.show()  # Optionally, comment this out if running in a non-interactive environment

# Save ANOVA results to a DataFrame and then to a CSV file
anova_pred_df = pd.DataFrame([(k, v.statistic, v.pvalue) for k, v in anova_pred_results.items()], columns=['Feature', 'F-Statistic', 'P-Value'])
anova_pred_df.to_csv('./correlation_results/OWANOVA_pred_pers_against_criteria.csv', index=False)



# Load the predicted persuasiveness scores (assuming it's already loaded and processed earlier in your code)
predicted_scores_path = './results/persuasiveness/1/mistral-1/dimentions_all_files_persuasiveness_evaluation_prompt_1_test_1_LLM_answers_mistral_7b.csv'
predicted_scores = pd.read_csv(predicted_scores_path)
predicted_scores['score'] = predicted_scores['score'].astype(float)

# Merge llm_features with predicted_scores on 'ID'
merged_llm_pred = pd.merge(llm_features, predicted_scores[['ID', 'score']], on='ID')

# Drop 'ID' column for correlation calculation
merged_llm_pred = merged_llm_pred.drop(columns='ID')

# Compute Spearman and Pearson correlations
llm_pred_spearman_corr = merged_llm_pred.corr(method='spearman')['score'].drop('score')
llm_pred_pearson_corr = merged_llm_pred.corr(method='pearson')['score'].drop('score')

# Save the correlation results to CSV files
llm_pred_spearman_corr.to_csv('./correlation_results/LLM_crit_against_pred_pers_Spearman.csv')
llm_pred_pearson_corr.to_csv('./correlation_results/LLM_crit_against_pred_pers_Pearson.csv')
