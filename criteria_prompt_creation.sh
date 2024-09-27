#!/bin/bash

# Load variables from the configuration file
source ./analysis_configuration.sh

# Create a directory for prompts if it doesn't exist
mkdir -p prompts


# Read content from the task file, removing any carriage return characters
task_content=$(tr -d '\r' < "$task_folder/criteria_tasks/$criteria_category/$criteria/$prompt_version.txt" | tr -d '\n')

# Use null-terminated output from find to handle file names with spaces
find "$data_folder/$dataset/transcripts/$clip" -type f -name "*.txt" -print0 | while IFS= read -r -d '' transcript_file; do

    # Extract information from the transcript file path
    relative_path=$(dirname "${transcript_file}")
    relative_path_without_prefix=${relative_path#$transcripts_folder/}  # Remove the prefix
    transcript_base=$(basename "$transcript_file" .txt)


    # Combine texts from task_file and the current transcriptXXX-&&-category.txt file
    constructed_prompt="Analysis description: $task_content TRANSCRIPT = [ $(cat "$transcript_file" | tr -d '\r' | tr -d '\n') ] "

    # Create subdirectories in prompts folder based on the structure of transcripts_whisper/
    output_folder="prompts/$dataset/$criteria_category/$criteria/$prompt_version/$clip/" # Updated output folder path
    mkdir -p "$output_folder"

    # Save the constructed prompt to prompts/promptXXX-category.txt
    output_file="$output_folder/$transcript_base.txt"  # Remove spaces from transcript_base
    echo "$constructed_prompt" > "$output_file"
    echo "Constructed prompt saved to: $output_file"
done
