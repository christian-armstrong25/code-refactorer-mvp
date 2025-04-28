import os
import json

def get_python_files_content(directory):
    # Dictionary to store filename as key and file content as value
    python_files_content = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a Python file
        if filename.endswith('.py'):
            file_path = os.path.join(directory, filename)
            # Open and read the file content
            with open(file_path, 'r') as file:
                file_content = file.read()
                # Add to dictionary
                python_files_content[filename] = file_content

    return python_files_content

# Example usage
directory_path = '/Users/typham-swann/Downloads/Repo_Cleaner/src'
python_files_content = get_python_files_content(directory_path)

# Output the dictionary to a JSON file
output_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/python_files_content.json'
with open(output_path, 'w') as json_file:
    json.dump(python_files_content, json_file, indent=4)

print(f"Python files content has been written to {output_path}")


