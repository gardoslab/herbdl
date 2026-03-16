##Initial Skeleton Code

import os
import json
import shutil

# Function to read and parse the JSON file
def read_json(json_path):
    with open(json_path, 'r') as file:
        d = json.load(file)
        data = d["log_history"]
    return data

# Function to sort the data by loss
def sort_by_loss(data):
    print(type(data[0]))
    for i in range(len(data)):
        if "loss" not in data[i]:
            del data[i]
            break

    return sorted(data, key=lambda x: x['loss'])

# Function to process directories based on the sorted data
def process_directories(data, base_path):
    # Get the top 10 entries with the lowest losses
    top_10 = data[:10]
    print({entry['loss'] for entry in top_10})
    top_steps = {entry['step'] for entry in top_10}

    # List all directories in the base path
    all_dirs = os.listdir(base_path)

    # Identify directories to keep and delete
    dirs_to_keep = {f"checkpoint-{step}" for step in top_steps}
    dirs_to_delete = [d for d in all_dirs if d.startswith('checkpoint-') and d not in dirs_to_keep]

    # Delete the directories not in the top 10
    for dir_name in dirs_to_delete:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted directory: {dir_path}")

    print("Processing complete. Kept directories:", dirs_to_keep)

# Main function
def main(json_path, base_path):
    # Read and sort the data
    data = read_json(json_path)
    sorted_data = sort_by_loss(data)

    # Process the directories
    process_directories(sorted_data, base_path)

# Example usage
if __name__ == "__main__":
    json_path = "../output/retraining/checkpoint-302000/trainer_state.json"  # Replace with the actual path to your JSON file
    base_path = "../output/retraining"  # Replace with the actual path to your base directory
    main(json_path, base_path)
