from openai import OpenAI
import json
client = OpenAI(api_key="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")

def ai_helper_get_code_json(content_object, notebook_plan):
    filepaths = list(content_object.keys())
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You will be provided a plan for creating a notebook, and a project directory JSON. Please create a list of files necessary to create the notebook."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Plan: ```{notebook_plan}```\n\nPython Content JSON: ```{content_object}```"
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "relevent_pilepaths",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "filepaths": {
                "type": "array",
                "description": "List of relevant filepaths",
                "items": {
                "type": "string",
                "enum": filepaths
                }
            }
            },
            "required": [
            "filepaths"
            ],
            "additionalProperties": False
        }
        }
    }
    )

    needed_files = response.choices[0].message.content
    return needed_files

def helper_filter_code_fields(content_object, files_to_keep=None):
    # If files_to_keep is None, set it as an empty list to filter out all 'code' fields
    if files_to_keep is None:
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

def ai_helper_create_notebook(content_object, notebook_plan):
    content_object = helper_filter_code_fields(content_object, ai_helper_get_code_json(content_object, notebook_plan))
    response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "text": f"Python Contents Object: ```{content_object}```\n\nNotebook Plan: ```{notebook_plan}```\n\nPlease create a full jupiter notebook using the provided code and plan. You response should work entirely standalone as a turnkey notebook for the directory and be formatted in a way that it can directly be converted in a ipynb file. Your audience is another researcher who is picking up the directory to add to. The notebook should not be chatty or educational but straightforward and informative. Please make it clear where the code and markdown sections are",
            "type": "text"
            }
        ]
        }
    ]
    )

    notebook_string = response.choices[0].message.content
    return notebook_string


# FOR TESTING BELOW

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def save_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def load_text(filepath):
    with open(filepath, 'r') as file:
        return file.read()
    
content_object = load_json("/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/fix_differences/python_content.json")
    
# Load the notebook plan from a txt file
notebook_plan_filepath = "/Users/typham-swann/Desktop/research_code_app/python_logic/helper_functions/notebook_helpers/test_outputs/notebook_output.txt"  # Replace this path with the actual file path
notebook_plan = load_text(notebook_plan_filepath)

notebook_output = ai_helper_create_notebook(content_object, notebook_plan)

# Save the result to a new txt file
output_filepath = "/Users/typham-swann/Desktop/research_code_app/python_logic/helper_functions/notebook_helpers/test_outputs/notebook_string.txt"
with open(output_filepath, 'w') as file:
    file.write(notebook_output)
