import os
import json

def handle_files(directory):
    # Dictionary to store relative filepath as key and file content as value
    python_files_content = {}
    
    # Function to construct the filetree in JSON format
    def construct_file_tree(directory, root):
        file_tree = {
            "name": os.path.basename(directory),
            "is_directory": True,
            "children": []
        }
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Recursively construct file tree for directories
                file_tree["children"].append(construct_file_tree(item_path, root))
            else:
                # Append file information to the file tree
                file_tree["children"].append({
                    "name": item,
                    "is_directory": False,
                    "children": []
                })
                # If it's a Python file, read its content
                if item.endswith('.py'):
                    relative_path = os.path.relpath(item_path, root)
                    with open(item_path, 'r') as file:
                        python_files_content[relative_path] = file.read()
        return file_tree

    # Construct the file tree and ensure the relative paths are from the root
    file_tree_structure = construct_file_tree(directory, directory)

    return python_files_content, file_tree_structure

# Example usage
directory_path = '/Users/typham-swann/Desktop/research_code_app/python_logic_tester'
python_files_content, file_tree_structure = handle_files(directory_path)

# Output the python files content dictionary to a JSON file
output_content_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/handle_files/python_content.json'
with open(output_content_path, 'w') as json_file:
    json.dump(python_files_content, json_file, indent=4)

# Output the file tree structure to a JSON file
output_tree_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/handle_files/filetree.json'
with open(output_tree_path, 'w') as json_file:
    json.dump(file_tree_structure, json_file, indent=4)

print(f"Python files content has been written to {output_content_path}")
print(f"File tree structure has been written to {output_tree_path}")
