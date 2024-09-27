#!/bin/bash

# Load variables from the configuration file
source ./analysis_configuration.sh

# Create a directory for terminal output if it doesn't exist
mkdir -p terminaloutput

# Before your loop starts, define a way to read the ID from the list_ID.csv
# We use awk to extract the ID based on matching category, city, and transcript ID

# Check if there are prompt files in prompts/ directory and its subfolders
prompt_files=($(find prompts/$dataset/$dimension/$prompt_version/$clip/ -type f -name "*.txt"))
# echo $prompt_files
if [ ${#prompt_files[@]} -gt 0 ]; then
    # Iterate over found prompt files
    for prompt_file in "${prompt_files[@]}"; do
        # Extract relevant information from the prompt file path
        # relative_path=$(dirname "${prompt_file}")
        # echo $relative_path
        # relative_path_without_prefix=${relative_path#prompts/}  # Remove the prefix
        # echo $relative_path_without_prefix
        prompt_name=$(basename "$prompt_file" .txt | sed 's/^prompt//')
        # echo $prompt_name

        # # Extract category and city from the relative path
        # category=$(basename "$relative_path_without_prefix")
        # echo $category
        # city=$(basename "$(dirname "$relative_path_without_prefix")")
        # echo $city


        # Removing the category suffix from relative_path_without_prefix
        # relative_path_without_prefix=${relative_path_without_prefix%/*}
        # relative_path_without_prefix=${relative_path_without_prefix%/*}
        # echo $relative_path_without_prefix

        output_id=$prompt_name  # Change prompt to output prefix
        
        # Call the Python script to get the ID
        # id_from_csv=$(python3 ./preprocessing/read_id_from_csv.py "./$data_folder/$dataset/list_ID.csv" "$clip" "$city" "$output_id")
        
        # Call the Python script to check if the ID is accepted
        # Use it in case when you want to consider only sertain set of samples, specify this set in the .csv filr
        # is_id_accepted=$(python3 ./preprocessing/check_id_in_list.py "$id_from_csv" "your_accepted_samples.csv")

        prompt_content=$(tr -d '\r' < "$prompt_file" | tr -d '\n')

        output_folder="$annotation_results_folder/model_raw_output/$dataset/$dimension/$prompt_version/$model_name-$test_number/$clip/"
        mkdir -p "$output_folder"
        echo $output_folder

        #if ["$is_id_accepted" == "yes" ]; then
        echo "{\"model\": \"$model_name\", \"prompt\": \"Combining texts: $prompt_content\", \"stream\": false}" > payload.json
        curl -H "Content-Type: application/json" -X POST -d @payload.json "http://localhost:11434/api/generate" > "$output_folder/$prompt_name.json"
        rm payload.json
        # fi

    done
else
    echo "No prompt files found in prompts/ directory or its subdirectories."
fi
