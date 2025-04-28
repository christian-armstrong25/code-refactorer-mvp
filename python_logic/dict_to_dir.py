import os
import json

def write_python_files_from_json(json_path, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Load the JSON data
    with open(json_path, 'r') as json_file:
        python_files_content = json.load(json_file)

    # Iterate over each entry in the JSON
    for filename, content in python_files_content.items():
        # Create the full path for each file in the output directory
        file_path = os.path.join(output_directory, filename)

        # Write the content to the Python file
        with open(file_path, 'w') as file:
            file.write(content)

    print(f"Python files have been written to the directory: {output_directory}")

# Example usage
json_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/python_files_content.json'  # Path to the JSON file created earlier
output_directory = '/Users/typham-swann/Desktop/research_code_app/python_logic/output_dir'  # Directory to store Python files

write_python_files_from_json(json_path, output_directory)
