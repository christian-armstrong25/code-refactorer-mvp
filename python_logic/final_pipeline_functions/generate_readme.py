import os

def helper_check_readme(python_content, file_path):
    readme_path = os.path.join(file_path, 'README.md')
    if os.path.isfile(readme_path):
        with open(readme_path, 'r') as file:
            readme_content = file.read()
        return helper_generate_readme_text(python_content, readme_content)
    else:
        return helper_generate_readme_text(python_content, readme_content=None)

def helper_generate_readme_text(python_content, readme_content):
    if readme_content:
        
        return python_content
    else:
        return python_content

def helper_filter_code_fields_readme(content_object):
    # If files_to_keep is None, set it as an empty list to filter out all 'code' fields
    files_to_keep = []

    filtered_content = {}
    for filepath, details in content_object.items():
        # Copy the details dictionary to avoid mutating the original content_object
        filtered_details = details.copy()

        # If the filepath is not in the files_to_keep list, remove the 'code' field
        if filepath not in files_to_keep:
            filtered_details.pop("code", None)
        
        # Remove 'referenced_filepaths' and 'updated_references' for all files
        filtered_details.pop("referenced_filepaths", None)
        filtered_details.pop("updated_references", None)

        filtered_content[filepath] = filtered_details
    
    return filtered_content