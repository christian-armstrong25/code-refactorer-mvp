import json
import os

def create_summarized_filetree(filetree, content_object):
    file_list = []

    def process_node(node, path_so_far, is_root=False):
        result = {"name": node["name"]}

        # For the root node, do not include its name in the path
        if is_root:
            new_path = path_so_far
        else:
            new_path = path_so_far + node["name"] if path_so_far else node["name"]

        if node.get("is_directory", False):
            # Process child nodes if they exist
            children = node.get("children", [])
            processed_children = []
            for child in children:
                # For directories, add '/' to path
                child_path_so_far = new_path + '/' if new_path else ''
                processed_child = process_node(child, child_path_so_far, is_root=False)
                if processed_child:
                    processed_children.append(processed_child)
            if processed_children:
                result["children"] = processed_children
        else:
            # Add to file list for non-directory nodes (files)
            file_list.append(new_path)

            # Add summaries for Python files if available
            if node["name"].endswith('.py'):
                content_path = new_path
                if content_path in content_object:
                    content = content_object[content_path]
                    result["file_summary"] = content.get("file_summary")
                    result["function_information"] = content.get("function_information")
        return result

    # Start processing from the root node, setting is_root=True
    summarized_tree = process_node(filetree, "", is_root=True)

    # Return both the summarized filetree and the file list as a string
    file_list_string = "\n".join(file_list)
    return summarized_tree, file_list_string

# FOR TESTING ONLY BELOW THIS

def save_file_list_to_txt(file_list, filepath):
    """Helper function to save a list of files to a .txt file."""
    try:
        with open(filepath, 'w') as file:
            # Join the file paths into a single string, separated by spaces
            file.write("".join(file_list))
        print(f"File list saved to {filepath}")
    except IOError as e:
        print(f"Error saving file: {e}")


def load_json_from_file(filepath):
    """Helper function to load a JSON object from a file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    else:
        print(f"Error: File {filepath} does not exist.")
        return None

def pretty_print_json(data):
    """Helper function to print JSON data in a pretty format."""
    print(json.dumps(data, indent=4))

def save_json_to_file(data, filepath):
    """Helper function to save a JSON object to a file."""
    try:
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Summarized filetree saved to {filepath}")
    except IOError as e:
        print(f"Error saving file: {e}")

filetree_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/handle_files/filetree.json'
content_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/identify_references/python_content.json'
output_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/clean_filetree/summarized_filetree.json'
output_file_list_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/clean_filetree/file_list.txt'

# Load the JSON files
filetree = load_json_from_file(filetree_path)
content = load_json_from_file(content_path)

# Check if both JSON files were successfully loaded
if filetree and content:
    # Create the summarized filetree and get the file list
    summarized_filetree, file_list = create_summarized_filetree(filetree, content)

    # Save the summarized filetree as a JSON file
    save_json_to_file(summarized_filetree, output_path)

    # Save the list of files to a .txt file
    save_file_list_to_txt(file_list, output_file_list_path)
else:
    print("Failed to load one or both JSON files.")

