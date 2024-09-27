import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load llm_features
llm_features_path = './LLM_features/LLM_features.csv'
llm_features = pd.read_csv(llm_features_path)

# Define lexical feature categories
lex_categories = ['text_SD_LIWC']

# Prepare a DataFrame to store the best results for each LLM feature across categories
top_correlations = pd.DataFrame()

# Iterate through each lexical feature category
for lex_category in lex_categories:
    print(f"Processing lexical category: {lex_category}")
    # Load lexical features for the category
    lexical_features_path = f'./lexical_features/{lex_category}_full.csv'
    lexical_features = pd.read_csv(lexical_features_path)

    # Merge llm_features and lexical_features on 'ID'
    merged_data = pd.merge(llm_features, lexical_features, on='ID', how='inner')
    print(merged_data.head())
    # Drop rows with NaN values
    merged_data = merged_data.dropna()  
    # Drop 'ID' column for correlation calculation
    merged_data = merged_data.drop(columns=['ID'])

    # Calculate Spearman correlation between llm_features and lexical_features
    for llm_feature in llm_features.columns.drop('ID'):
        best_corr = 0.0  # Initialize to the lowest possible correlation value
        best_pvalue = float('inf')
        best_feature = None
        for lex_feature in lexical_features.columns.drop('ID'):
            spearman_corr = spearmanr(merged_data[llm_feature], merged_data[lex_feature])
            print(spearman_corr)
            # print(spearman_corr.correlation)
            # print(abs(spearman_corr.correlation))

            if spearman_corr.correlation is not None and abs(spearman_corr.correlation) >= best_corr:
                if (spearman_corr.correlation == best_corr):
                    if (spearman_corr.pvalue < best_pvalue):
                        best_corr = spearman_corr.correlation
                        best_pvalue = spearman_corr.pvalue
                        best_feature = lex_feature
                else:
                        best_corr = spearman_corr.correlation
                        best_pvalue = spearman_corr.pvalue
                        best_feature = lex_feature
                    

        # Compile the best results into the top_correlations DataFrame
        if best_feature is not None:
            top_correlations.loc[llm_feature, f'{lex_category}_Best_Feature&'] = f"{best_feature}&"
            top_correlations.loc[llm_feature, f'{lex_category}_Best_Correlation&'] = f"${round(best_corr, 2)}$&"
            top_correlations.loc[llm_feature, f'{lex_category}_Best_p-value&'] = f"${round(best_pvalue, 3)}$&"

# Save the compiled results to CSV file
top_correlations.to_csv('./correlation_results/crit_vs_lex/test_SD_LIWC.csv')
