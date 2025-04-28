from openai import OpenAI
client = OpenAI(api_key="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")
import json


def identify_references(code_dict):
    # Create a new dictionary to store the original code with function information added
    updated_code_dict = {}

    for file_path, code in code_dict.items():
        # Get the function summaries for the current file using the ai_summarize_functions function
        try:
            referenced_filepaths = ai_identify_references(code["code"])  # Summarize the functions
        except Exception as e:
            referenced_filepaths = {"error": str(e)}  # Handle any errors gracefully

        # Add the function information to the original code
        updated_code_dict[file_path] = {
            "code": code["code"],
            "function_information": code['function_information'],
            "file_summary": code['file_summary'],
            "referenced_filepaths": referenced_filepaths
        }

    updated_code_dict = helper_filter_generic_imports(updated_code_dict)

    return updated_code_dict

def ai_identify_references(python_text):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are tasked with identifying filepaths within a python file. You will see python code from a python file. You must respond with all external filepaths. It's important that you included all external filepaths identified. Be sure to include external and internal files, using the exact path used in the code. Include only filepaths themselves. Ignore variables or parameters that are not clear instances of a specific filepath mention."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom helpers.helper import preprocess_data, calculate_summary_statistics\n\n# Load external data (CSV file)\ndef load_data(file_path):\n    \"\"\"\n    Load CSV data into a pandas DataFrame.\n    \n    Parameters:\n        file_path (str): The path to the CSV file.\n        \n    Returns:\n        pd.DataFrame: DataFrame containing the loaded data.\n    \"\"\"\n    try:\n        data = pd.read_csv(file_path)\n        return data\n    except FileNotFoundError:\n        print(f\"Error: The file {file_path} was not found.\")\n        return None\n\n# Main analysis function\ndef analyze_data(file_path):\n    \"\"\"\n    Perform data analysis on the dataset.\n    \n    Parameters:\n        file_path (str): The path to the CSV file.\n    \"\"\"\n    # Step 1: Load the data\n    data = load_data(file_path)\n    if data is None:\n        return\n    \n    # Step 2: Preprocess the data (from helper.py)\n    processed_data = preprocess_data(data)\n    \n    # Step 3: Calculate summary statistics (from helper.py)\n    summary_stats = calculate_summary_statistics(processed_data)\n    print(\"Summary Statistics:\")\n    print(summary_stats)\n    \n    # Step 4: Plotting\n    plt.figure(figsize=(10, 6))\n    plt.hist(processed_data['column_of_interest'], bins=30, alpha=0.7)\n    plt.title('Distribution of Column of Interest')\n    plt.xlabel('Value')\n    plt.ylabel('Frequency')\n    plt.show()\n\nif __name__ == '__main__':\n    # Provide the path to your CSV file\n    file_path = 'data.csv'\n    \n    # Run the data analysis\n    analyze_data(file_path)"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "{\"referenced_filepaths\":[{\"filepath\":\"data.csv\"}],\"imported_filepaths\":[{\"import_filepath\":\"helpers.helper\"}]}"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": python_text
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=4095,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "filepath_references",
        "strict": True,
        "schema": {
            "type": "object",
            "required": [
            "referenced_filepaths",
            "imported_filepaths"
            ],
            "properties": {
            "referenced_filepaths": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "filepath"
                ],
                "properties": {
                    "filepath": {
                    "type": "string"
                    }
                },
                "additionalProperties": False
                }
            },
            "imported_filepaths": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "import_filepath"
                ],
                "properties": {
                    "import_filepath": {
                    "type": "string",
                    "description": "The path or module of the imported Python file"
                    }
                },
                "additionalProperties": False
                }
            }
            },
            "additionalProperties": False
        }
        }
    }
    )
    text = response.choices[0].message.content
    obj = json.loads(text)

    return obj

def helper_filter_generic_imports(directory):
    # Create a set of all possible module name suffixes from the directory file paths
    module_name_suffixes = set()
    for filepath in directory.keys():
        # Only consider .py files
        if filepath.endswith('.py'):
            # Remove .py extension
            module_path = filepath[:-3]
            # Replace / and \ with . to get module name
            module_name = module_path.replace('/', '.').replace('\\', '.')
            # Split the module name into parts
            parts = module_name.split('.')
            # Generate all possible suffixes
            for i in range(len(parts)):
                suffix = '.'.join(parts[i:])
                module_name_suffixes.add(suffix)

    # Now, for each file in the directory
    for file_info in directory.values():
        referenced = file_info.get('referenced_filepaths', {})
        imported_filepaths = referenced.get('imported_filepaths', [])
        # We will build a new list of imported_filepaths that could be internal
        new_imported_filepaths = []
        for import_entry in imported_filepaths:
            import_filepath = import_entry.get('import_filepath')
            if import_filepath in module_name_suffixes:
                new_imported_filepaths.append(import_entry)
        # Update the imported_filepaths list
        referenced['imported_filepaths'] = new_imported_filepaths
        # Update the file's referenced_filepaths
        file_info['referenced_filepaths'] = referenced

    return directory


# Sample usage

# Load the JSON file
input_json_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/summarize_file/python_content.json'  # Update with the correct path to the input file
output_json_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/identify_references/python_content.json'  # Update with the desired output file path

try:
    with open(input_json_path, 'r') as f:
        code_dict = json.load(f)  # Load the JSON as a Python dictionary
except Exception as e:
    print(f"Error loading input JSON: {str(e)}")
    code_dict = {}

# Call the summarize_functions function on the loaded code_dict
updated_code_dict = identify_references(code_dict)

# Write the output to a new JSON file
try:
    with open(output_json_path, 'w') as f:
        json.dump(updated_code_dict, f, indent=4)  # Output the result to a new JSON file
    print(f"Output saved to {output_json_path}")
except Exception as e:
    print(f"Error saving output JSON: {str(e)}")
