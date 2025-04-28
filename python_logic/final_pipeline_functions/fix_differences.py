import copy
from pathlib import PurePosixPath
import json
import os
import re

def fix_differences(content_object, filetree_map_object):
    updated_content_object = helper_determine_replacements(content_object, filetree_map_object)
    updated_content_object = helper_update_code(updated_content_object)
    updated_content_object = helper_update_content_keys(updated_content_object, filetree_map_object)
    return updated_content_object

def helper_determine_replacements(content_object, filetree_map_object):
    # Define the project root directory (base directory)
    project_root = "/Users/typham-swann/Desktop/research_code_app"
    base_dir_name = "python_logic_tester"
    base_dir = os.path.join(project_root, base_dir_name)

    # Build a mapping from absolute original paths to absolute new paths
    original_to_new_paths = {}
    for original_rel_path, new_rel_path in filetree_map_object.items():
        original_abs_path = os.path.normpath(os.path.join(project_root, original_rel_path))
        new_abs_path = os.path.normpath(os.path.join(project_root, new_rel_path))
        original_to_new_paths[original_abs_path] = new_abs_path

    # Initialize the updated content object
    updated_content_object = {}

    # Process each file in the content object
    for file_rel_path, file_data in content_object.items():
        # Get original and new absolute paths of the file
        original_file_abs_path = os.path.normpath(os.path.join(base_dir, file_rel_path))
        new_file_abs_path = original_to_new_paths.get(original_file_abs_path, original_file_abs_path)

        # Initialize the updated references list
        updated_references = []

        # Get referenced filepaths
        referenced_filepaths = file_data.get('referenced_filepaths', {})
        for ref_type in ['referenced_filepaths', 'imported_filepaths']:
            ref_list = referenced_filepaths.get(ref_type, [])
            for ref in ref_list:
                original_ref_path = ref.get('filepath')
                if not original_ref_path:
                    continue

                # Normalize the original referenced path
                original_ref_abs_path = os.path.normpath(original_ref_path)
                if not original_ref_abs_path.startswith(project_root):
                    # If the path is not under project_root, skip it
                    continue

                # Get the relative path from project root
                original_ref_rel_path = os.path.relpath(original_ref_abs_path, project_root)
                original_ref_abs_path = os.path.join(project_root, original_ref_rel_path)

                # Get the new absolute path from the filetree map
                new_ref_abs_path = original_to_new_paths.get(original_ref_abs_path, original_ref_abs_path)

                # Compute the new relative path from the new file location to the new referenced file location
                new_relative_path = os.path.relpath(new_ref_abs_path, os.path.dirname(new_file_abs_path))

                # Add to updated_references
                updated_references.append({
                    'original_filepath': original_ref_path,
                    'new_filepath': new_relative_path
                })

        # Add 'updated_references' to the file data
        updated_file_data = dict(file_data)
        updated_file_data['updated_references'] = updated_references

        # Update the content object with the new data
        updated_content_object[file_rel_path] = updated_file_data

    return updated_content_object

def helper_update_code(content_object):
    updated_content_object = {}

    for file_rel_path, file_data in content_object.items():
        code = file_data.get('code', '')
        updated_references = file_data.get('updated_references', [])

        # Make a copy of the code to perform replacements
        updated_code = code

        for ref in updated_references:
            original_filepath = ref.get('original_filepath')
            new_filepath = ref.get('new_filepath')

            if not original_filepath or not new_filepath:
                continue

            # Escape special characters in filepaths for regex
            escaped_original_filepath = re.escape(original_filepath)

            # Use regex to replace all occurrences of the old filepath with the new filepath
            updated_code = re.sub(escaped_original_filepath, new_filepath, updated_code)

        # Update the code in the file data
        updated_file_data = dict(file_data)
        updated_file_data['code'] = updated_code

        # Update the content object with the new file data
        updated_content_object[file_rel_path] = updated_file_data

    return updated_content_object

def helper_update_content_keys(content_object, filetree_map):
    if not filetree_map:
        return content_object  # No mapping to update

    # Extract base path from the first key in the filetree map
    first_old_path = next(iter(filetree_map))
    first_content_key = next(iter(content_object))

    # Determine the base path by removing the content object key from the filetree map key
    if first_content_key in first_old_path:
        index = first_old_path.rfind(first_content_key)
        base_path = first_old_path[:index]
    else:
        raise ValueError("Cannot determine base path from filetree map and content object keys.")

    # Create a mapping from old relative paths to new relative paths
    old_to_new_relative_paths = {}
    for old_full_path, new_full_path in filetree_map.items():
        # Get the relative paths by removing the base path
        if old_full_path.startswith(base_path):
            old_relative_path = old_full_path[len(base_path):]
            new_relative_path = new_full_path[len(base_path):]
            old_to_new_relative_paths[old_relative_path] = new_relative_path
        else:
            # Skip paths that do not match the base path
            continue

    # Update the content object keys using the mapping
    new_content_object = {}
    for key, value in content_object.items():
        if key in old_to_new_relative_paths:
            new_key = old_to_new_relative_paths[key]
            new_content_object[new_key] = value
        else:
            # If the key is not in the mapping, keep it as is
            new_content_object[key] = value

    return new_content_object

# /**************/
# FOR TESTING BELOW THIS
# /**************/

# Sample paths for input and output JSON files
content_object_path = "/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/identify_references/python_content.json"
filetree_map_path = "/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/structure_filetree/filetree_map.json"
output_path = "/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/fix_differences/python_content.json"

# Load sample Content Object and Filetree Map Object from JSON files
with open(content_object_path, "r") as content_file:
    content_object = json.load(content_file)

with open(filetree_map_path, "r") as filetree_file:
    filetree_map_object = json.load(filetree_file)

# Run the update_content_object function
updated_content_object = fix_differences(content_object, filetree_map_object)

# Output the updated Content Object to a new JSON file
with open(output_path, "w") as output_file:
    json.dump(updated_content_object, output_file, indent=4)

print(f"Updated Content Object saved to {output_path}")
