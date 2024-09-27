# config.sh

# Main Variables 
export dataset="3MTFrench"
export model_name="llama2"
export dimension="persuasiveness"
export criteria_category="storytelling"
export criteria="metaphor"
export clip="full"
# Prompt Attempts
export prompt_version="1"
export test_number="1"
# Main folders
export task_folder="./tasks" # Tasks for the prompts
export data_folder="./data" # Datasets with transcripts and labels
export feature_folder="./features" # Lexical features and Criteria
export prompt_folder="./prompts" # Prompts created for each transcript
export annotation_results_folder="./annotation_results" # Model raw outputs and extracted scores
