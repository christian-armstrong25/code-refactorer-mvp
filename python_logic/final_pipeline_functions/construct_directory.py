import os
import json
import shutil
import nbformat  # Import nbformat for notebook handling
from pathlib import Path

def construct_directory(original_dir, content_object, new_filetree, output_dir, notebook_object):
    paths_map = helper_create_path_mapping(original_dir, new_filetree)
    helper_build_directory(original_dir, content_object, new_filetree, paths_map, output_dir)
    helper_add_notebook(notebook_object, output_dir, new_filetree)  # Pass new_filetree to locate project root

def helper_create_path_mapping(base_dir, filetree):
    def get_existing_files(directory):
        """Returns list of all files (not directories) under the given directory."""
        existing_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                # Get full absolute path
                full_path = os.path.join(root, file)
                # Get path relative to base_dir
                rel_path = os.path.relpath(full_path, directory)
                existing_files.append(rel_path)
        return existing_files

    def get_filetree_paths(node, current_path=""):
        """Extracts all file paths from the filetree JSON structure."""
        paths = []
        new_path = os.path.join(current_path, node["name"])

        if not node["is_directory"]:
            # Do not include base_dir in the path
            paths.append(new_path)

        for child in node.get("children", []):
            paths.extend(get_filetree_paths(child, new_path))

        return paths

    def find_matching_file(original_file, filetree_paths):
        """
        Finds the matching new path for an original file based on filename.
        Returns None if no match is found.
        """
        filename = os.path.basename(original_file)
        matches = [p for p in filetree_paths if os.path.basename(p) == filename]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # If multiple matches, try to match based on file type/extension
            exact_matches = [p for p in matches if p.endswith(os.path.splitext(filename)[1])]
            return exact_matches[0] if exact_matches else matches[0]
        return None

    # Get list of existing files with relative paths
    existing_files = get_existing_files(base_dir)

    # Get list of paths from filetree (relative paths)
    filetree_paths = get_filetree_paths(filetree)

    # Create mapping
    path_mapping = {}
    for orig_path in existing_files:
        new_path = find_matching_file(orig_path, filetree_paths)
        if new_path:
            path_mapping[orig_path] = new_path

    return path_mapping

def helper_build_directory(old_directory_path, content_object, filetree_object, paths_map, output_directory_path):
    # Function to recursively create directories as per the filetree object
    def create_directories(tree_node, current_path):
        if tree_node.get('is_directory', False):
            dir_name = tree_node['name']
            dir_path = os.path.join(current_path, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            for child in tree_node.get('children', []):
                create_directories(child, dir_path)
    
    # Create the new directory structure in the output directory
    create_directories(filetree_object, output_directory_path)
    
    # Process each file in the paths map
    for old_file_rel_path, new_file_rel_path in paths_map.items():
        old_file_abs_path = os.path.join(old_directory_path, old_file_rel_path)
        new_file_abs_path = os.path.join(output_directory_path, new_file_rel_path)

        # Ensure the directory for the new file exists
        new_file_dir = os.path.dirname(new_file_abs_path)
        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)

        if new_file_abs_path.endswith('.py'):
            # It's a Python file; replace its content with the code from the content object
            filename = os.path.basename(old_file_rel_path)
            code_info = content_object.get(filename)
            if code_info:
                code_content = code_info.get('code', '')
                with open(new_file_abs_path, 'w') as file:
                    file.write(code_content)
            else:
                print(f"Warning: Code for {filename} not found in the content object. Copying original file.")
                if os.path.exists(old_file_abs_path):
                    shutil.copy2(old_file_abs_path, new_file_abs_path)
                else:
                    print(f"Error: Original file {old_file_abs_path} does not exist.")
        else:
            # For non-Python files, copy them to the new location without modification
            if os.path.exists(old_file_abs_path):
                shutil.copy2(old_file_abs_path, new_file_abs_path)
            else:
                print(f"Error: Original file {old_file_abs_path} does not exist.")

def helper_add_notebook(notebook_object, output_directory_path, filetree_object):
    """Adds a starter notebook to the root of the project directory."""
    # The root of the project is the top-level directory in the filetree
    project_root_name = filetree_object['name']
    project_root_path = os.path.join(output_directory_path, project_root_name)

    # Ensure the project root directory exists
    if not os.path.exists(project_root_path):
        os.makedirs(project_root_path)

    # Path to place the notebook
    notebook_path = os.path.join(project_root_path, 'starter_notebook.ipynb')

    # Create the notebook
    nb = nbformat.from_dict(notebook_object)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
# /********************
# FOR TESTING BELOW THIS
# /********************

# Read filetree from JSON file
with open('/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/run_pipeline_intermediate/organized_file_tree.json', 'r') as f:
    filetree = json.load(f)

with open('/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/run_pipeline_intermediate/python_files_content.json', 'r') as f:
    content_object = json.load(f)

with open('/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/generate_notebook/notebook_output.json', 'r') as f:
    notebook_obj = json.load(f)

old_directory = '/Users/typham-swann/Desktop/research_code_app/python_logic_tester/weather_sample_directory'

output_directory = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/construct_directory'

construct_directory(old_directory, content_object, filetree, output_directory, notebook_obj)
