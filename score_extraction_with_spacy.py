import pandas as pd
import spacy
import os
import subprocess

# Load the configuration from the .sh file
def load_config(file_path):
    command = f"bash -c 'source {file_path} && env'"
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    env_vars = {}
    for line in proc.stdout:
        key, _, value = line.decode('utf-8').partition("=")
        env_vars[key.strip()] = value.strip()
    proc.communicate()
    return env_vars

# Load the configuration variables
config_file = './config.sh'
config_vars = load_config(config_file)

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Function to extract numerical entities from text
def extract_numerical_entities(text):
    doc = nlp(text)
    numerical_entities = []
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":  # Check if the entity is a cardinal number
            numerical_entities.append(ent.text)
    return numerical_entities

# Replace hardcoded variables with values from the config file
json_subfolder = config_vars.get('model_name')
dimension = config_vars.get('dimension')
criteria_group = config_vars.get('criteria_category')
criteria = config_vars.get('criteria')
version = config_vars.get('prompt_version')
test_number = config_vars.get('test_number')
model_name_with_test_number = f"{json_subfolder}-{test_number}"

# Path to the folder where the CSV file will be saved
csv_folder = os.path.join('./dimension_evaluation/results', dimension, version, model_name_with_test_number)
csv_filename = f'dimentions_all_files_{dimension}_evaluation_prompt_{version}_test_{test_number}_LLM_answers_{json_subfolder}_7b.csv'

# Paths to the answer and ID files
answers_file_path = os.path.join(csv_folder, csv_filename)
ids_file_path = "./dimension_evaluation/list_ID.csv"

# Read the CSV files
answers_df = pd.read_csv(answers_file_path, header=0)
ids_df = pd.read_csv(ids_file_path, header=0, sep=",")
print(answers_df.columns)
print(ids_df.columns)

# Function to retrieve ID from the list_ID.csv file
def get_id(transcript_id):
    row = ids_df[ids_df['transcript ID'] == transcript_id]
    if not row.empty:
        return row.iloc[0]['ID']
    else:
        return None

# Apply the function to extract numerical entities and create the "numerical information" column
answers_df["numerical information"] = answers_df["response"].apply(extract_numerical_entities)

# Extract the first CARDINAL to the "score" column
answers_df["score"] = answers_df["numerical information"].apply(lambda x: x[0] if x else None)

# Function to check the quality of the answer based on numerical information
def check_quality(numbers):
    if numbers:
        count_2_3_digit = sum(1 for num in numbers if num.isdigit() and len(num) in [2, 3])
        if count_2_3_digit <= 2:
            return 1
    return 0

# Apply the function to determine the quality of the answer and create the "quality of answer" column
answers_df["quality of answer"] = answers_df["numerical information"].apply(check_quality)

# Add ID column by mapping transcript IDs to IDs from list_ID.csv
answers_df["ID"] = answers_df["transcript ID"].apply(get_id)

# Save the updated DataFrame back to the CSV file
answers_df.to_csv(answers_file_path, index=False)

print("Extraction and saving completed.")
