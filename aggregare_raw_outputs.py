import os
import json
import csv
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
config_file = './analysis_configuration.sh'
config_vars = load_config(config_file)

# Path variables based on configuration
json_subfolder = config_vars.get('model_name')
annotation_results_folder = config_vars.get('annotation_results_folder')
dataset = config_vars.get('dataset')
clip = config_vars.get('clip')
dimension = config_vars.get('dimension')
criteria_group = config_vars.get('criteria_category')
criteria = config_vars.get('criteria')
version = config_vars.get('prompt_version')
test_number = config_vars.get('test_number')
model_name_with_test_number = f"{json_subfolder}-{test_number}"

# Paths to JSON and CSV folders
json_parent_folder = os.path.join(annotation_results_folder, 'model_raw_output', dataset, dimension, version, model_name_with_test_number, clip)
csv_folder = os.path.join(annotation_results_folder, 'extracted_annotation', dimension, version, model_name_with_test_number)
csv_filename = f'Evaluation_of_{dimension}_with_prompt_version_{version}_test_number_{test_number}_model_{json_subfolder}.csv'

# List all JSON files in the folder
json_files = []
for json_file in os.listdir(json_parent_folder):
    if json_file.endswith(".json"):
        json_files.append({
            "ID": os.path.splitext(json_file)[0],  # Save the name of the JSON file as the ID
            "clip": config_vars.get('clip'),       # Save the current clip variable
        })

# Check if there are any JSON files
if json_files:
    # Create the CSV file
    os.makedirs(csv_folder, exist_ok=True)
    csv_file_path = os.path.join(csv_folder, csv_filename)

    # Extract all unique fieldnames
    all_fieldnames = set()
    for json_data in json_files:
        with open(os.path.join(json_parent_folder, json_data["ID"] + ".json"), 'r') as jsonfile:
            try:
                data = json.load(jsonfile)
                all_fieldnames.update(data.keys())
            except json.JSONDecodeError:
                print(f"Empty JSON file: {json_data['ID']}.json")

    # Include additional fieldnames
    all_fieldnames.update(["ID", "clip"])

    # Open CSV file for writing
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames, extrasaction='ignore')

        # Write header
        writer.writeheader()

        # Iterate through JSON files
        for json_data in json_files:
            # Load JSON data
            with open(os.path.join(json_parent_folder, json_data["ID"] + ".json"), 'r') as jsonfile:
                try:
                    data = json.load(jsonfile)
                except json.JSONDecodeError:
                    print(f"Empty JSON file: {json_data['ID']}.json")
                    data = {}  # Set empty data if JSON file is empty

            # Add ID and clip to the data
            data.update(json_data)

            # Write data to CSV file
            writer.writerow(data)
else:
    print("No JSON files found.")
