from openai import OpenAI
import json
client = OpenAI(api_key="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")


def structure_filetree(content_json, file_list):
    response = client.chat.completions.create(
    model="o1-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Descriptive JSON: ```({'name': 'weather_sample_directory', 'children': [{'name': 'weather_reading.py', 'file_summary': \"The `weather_reading.py` script is designed to read weather data from CSV files for the years 2018, 2019, and 2020. It includes the `read_csv` function which opens a specified CSV file, reads its content using the CSV reader, and returns the data as a list of rows. It then uses this function to load data from three specified CSV files ('data/weather_2018.csv', 'data/weather_2019.csv', and 'data/weather_2020.csv'), storing the results in the variables `data_2018`, `data_2019`, and `data_2020`. The content of the data for 2018 and 2019 is printed for debugging. This script does not depend on other files in the project and acts independently for data reading purposes.\", 'function_information': [{'function_name': 'read_csv', 'function_description': 'Reads a CSV file from the specified file path and returns its content as a list of rows.'}]}, {'name': 'data_1', 'children': [{'name': 'weather.csv'}]}, {'name': 'weather_processing.py', 'file_summary': \"The `weather_processing.py` script reads CSV files containing weather data and processes this data to calculate and print the average temperature and humidity. It defines a function `process_weather`, which accepts a file path as an argument and reads the weather data, expecting temperatures in the second column and humidity in the third column. The script then calculates the average values of these parameters by summing them up and dividing by the number of records. It prints the results directly without returning them.\\n\\nThe script processes weather data from two files, 'data/weather_2018.csv' and 'data/weather_2019.csv'. The script relies on the `csv` module for handling CSV file operations.\\n\\nNo external functions from other scripts or additional functionalities from the 'utils.py' or 'stats_calculator.py' are used in this script, indicating that it operates independently within its scope.\", 'function_information': [{'function_name': 'process_weather', 'function_description': \"Processes a CSV file containing weather data to calculate and print the average temperature and humidity. The file is specified by 'file_path', and it is expected to have temperature and humidity data in the second and third columns, respectively, with a header that is skipped.\"}]}, {'name': 'utils.py', 'file_summary': 'The `utils.py` file defines a utility function, `to_float(value)`, which attempts to convert an input value to a float. If the conversion is successful, it returns the float value. Otherwise, it returns the string \"N/A\". This function can be useful for handling data conversions where the input might not always be in a numeric format, thus providing a default fallback (\"N/A\") when the conversion fails. Although the file doesn\\'t directly interact with other files, the `to_float` function could be referenced by other scripts in the project for consistent handling of potential numeric conversions, especially when dealing with data entries from CSV files which might not always be clean or well-formatted.', 'function_information': [{'function_name': 'to_float', 'function_description': \"Attempts to convert a given value to a float; returns the float value if successful, otherwise returns 'N/A'.\"}]}, {'name': 'README.txt'}, {'name': 'stats_calculator.py', 'file_summary': 'The `stats_calculator.py` script is designed to calculate and print the minimum and maximum temperatures from a given dataset. It contains two functions: \\n\\n1. `get_min_max(data)`: This function attempts to determine the minimum and maximum temperature from the input data, which it assumes has temperature values in the second element of each row. However, it does not properly convert these values to integers, which could lead to issues if data is not in the expected format.\\n\\n2. `calculate_statistics()`: This is a higher-level function that defines a static dataset directly within the function, calls `get_min_max` to compute the min and max temperatures, and then prints the results. The dataset used in this function has some incomplete entries and misformatted items, which could cause the calculations to behave unexpectedly. Furthermore, this script directly prints results rather than outputting them to a CSV or file, and it does not adequately handle instances where data might be missing or incomplete.\\n\\nThere are no external dependencies within the script itself, but it assumes it can interact with CSV files and real datasets, potentially involving files managed by other scripts like `weather_reading.py` or `weather_processing.py` for acquiring or processing data.', 'function_information': [{'function_name': 'get_min_max', 'function_description': 'Finds and returns the minimum and maximum temperature from a dataset. Assumes the second element in each row of data corresponds to the temperature.'}, {'function_name': 'calculate_statistics', 'function_description': \"Calculates and prints the minimum and maximum temperatures from a dataset. It calls 'get_min_max' function and handles the data directly within the same function.\"}]}, {'name': 'weather_stats_final.csv'}]}```\n\nList of Project Files: ```'weather_reading.py\\ndata_1/weather.csv\\nweather_processing.py\\nutils.py\\nREADME.txt\\nstats_calculator.py\\nweather_stats_final.csv'```\n\nYou are a codebase restructurer. You will be provided a directory JSON that contains the filetree and information about each python file. You should restructure the project using project organization best practices. Please make a new filetree that is maximally organized and intuitive so that a new user could download it and understand what's going on. It is important that the new project contains only and all of the files in the original. File names should remain the same. Please respond with only the new organized filetree. "
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "```\nweather_sample_directory/\n├── data/\n│   ├── raw/\n│   │   └── weather.csv\n│   └── processed/\n│       └── weather_stats_final.csv\n├── src/\n│   ├── weather_reading.py\n│   ├── weather_processing.py\n│   ├── stats_calculator.py\n│   └── utils.py\n├── README.txt\n```"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Descriptive JSON: ```{content_json}```\n\nList of Project Files: ```{file_list}```\n\nYou are a codebase restructurer. You will be provided a directory JSON that contains the filetree and information about each python file. You should restructure the project using project organization best practices. Please make a new filetree that is maximally organized and intuitive so that a new user could download it and understand what's going on. It is important that the new project contains only and all of the files in the original. File names should remain the same. Please respond with only the new organized filetree. "
            }
        ]
        }
    ]
    )
    text = response.choices[0].message.content

    return text

def format_filetree_text(filetree_string):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You are an expert file organizer. Given an existing file structure and summaries of each file, propose a new, more intuitive directory structure. It is important that all the file names remain unchanged and that all files are included. ",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "|-- uploads/\n|   |-- selected_few_shot_pipeline/\n|   |   |-- .DS_Store\n|   |   |-- __pycache__/\n|   |   |   |-- api.cpython-312.pyc\n|   |   |   |-- api.cpython-37.pyc\n|   |   |   |-- api.cpython-39.pyc\n|   |   |-- api.py\n|   |   |-- classifier.py\n|   |   |-- classifier_new_data.py\n|   |   |-- errors.csv\n|   |   |-- find_errors.py\n|   |   |-- intercoder_metrics.py\n|   |   |-- metrics.py\n|   |   |-- prompts.json\n|   |   |-- refactor_prompts.py\n|   |   |-- results/\n|   |   |   |-- results.csv\n|   |   |   |-- results_acot_150.csv\n|   |   |   |-- results_split_150.csv\n|   |   |   |-- results_split_150_old_prompts.csv\n|   |   |   |-- results_split_50.csv\n|   |   |   |-- results_split_500_old_prompts.csv\n|   |   |   |-- results_split_50_old_prompts.csv\n|   |   |-- results_split_400.csv\n|   |   |-- results_split_500.csv\n|   |   |-- results_splitcot_400.csv\n|   |   |-- results_test_3_few_4o.csv\n|   |   |-- results_test_ACoT_4o.csv\n|   |   |-- results_test_ind_cot.csv\n|   |   |-- results_test_manyshot_4o.csv\n|   |   |-- results_test_manyshot_optimized_4o.csv\n|   |   |-- results_test_manyshot_optimized_v1_4o.csv\n|   |   |-- results_test_manyshot_optimized_v2_4o.csv\n|   |   |-- results_test_manyshot_precise_seperate_delay_4o.csv\n|   |   |-- results_test_manyshot_precise_system_prompts_stronger_4o.csv\n|   |   |-- results_train_3_few_4o.csv\n|   |   |-- results_train_manyshot_4turbo.csv\n|   |   |-- results_train_manyshot_optimized_4o.csv\n|   |   |-- results_validation_3_few_4o.csv\n|   |   |-- results_validation_manyshot_4o.csv\n|   |   |-- results_validation_manyshot_all_claims_4o.csv\n|   |   |-- results_validation_manyshot_optimized_2check_4o.csv\n|   |   |-- results_validation_manyshot_optimized_4o.csv\n|   |   |-- results_validation_manyshot_optimized_4o_temp1.csv\n|   |   |-- results_validation_manyshot_optimized_v1_4o.csv\n|   |   |-- results_validation_manyshot_precise_seperate_delay_4o.csv\n|   |   |-- results_validation_manyshot_precise_system_prompts_4o.csv\n|   |   |-- results_validation_manyshot_precise_system_prompts_stronger_4o.csv\n|   |   |-- verify_new_prompts.py"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "text": "{\"name\":\"uploads\",\"is_directory\":true,\"children\":[{\"name\":\"selected_few_shot_pipeline\",\"is_directory\":true,\"children\":[{\"name\":\".DS_Store\",\"is_directory\":false,\"children\":[]},{\"name\":\"__pycache__\",\"is_directory\":true,\"children\":[{\"name\":\"api.cpython-312.pyc\",\"is_directory\":false,\"children\":[]},{\"name\":\"api.cpython-37.pyc\",\"is_directory\":false,\"children\":[]},{\"name\":\"api.cpython-39.pyc\",\"is_directory\":false,\"children\":[]}]},{\"name\":\"api.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"classifier.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"classifier_new_data.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"errors.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"find_errors.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"intercoder_metrics.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"metrics.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"prompts.json\",\"is_directory\":false,\"children\":[]},{\"name\":\"refactor_prompts.py\",\"is_directory\":false,\"children\":[]},{\"name\":\"results\",\"is_directory\":true,\"children\":[{\"name\":\"results.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_acot_150.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_150.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_150_old_prompts.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_50.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_500_old_prompts.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_50_old_prompts.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_400.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_split_500.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_splitcot_400.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_3_few_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_ACoT_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_ind_cot.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_optimized_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_optimized_v1_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_optimized_v2_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_precise_seperate_delay_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_test_manyshot_precise_system_prompts_stronger_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_train_3_few_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_train_manyshot_4turbo.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_train_manyshot_optimized_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_3_few_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_all_claims_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_2check_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_4o_temp1.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_optimized_v1_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_precise_seperate_delay_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_precise_system_prompts_4o.csv\",\"is_directory\":false,\"children\":[]},{\"name\":\"results_validation_manyshot_precise_system_prompts_stronger_4o.csv\",\"is_directory\":false,\"children\":[]}]},{\"name\":\"verify_new_prompts.py\",\"is_directory\":false,\"children\":[]}]}]}",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "text": filetree_string,
            "type": "text"
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=2612,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "directory_tree",
        "schema": {
            "type": "object",
            "required": [
            "name",
            "is_directory",
            "children"
            ],
            "properties": {
            "name": {
                "type": "string"
            },
            "children": {
                "type": "array",
                "items": {
                "$ref": "#"
                }
            },
            "is_directory": {
                "type": "boolean"
            }
            },
            "additionalProperties": False
        },
        "strict": True
        }
    }
    )
    text = response.choices[0].message.content
    obj = json.loads(text)

    return obj

# FOR TESTING ONLY BELOW THIS

# Load the JSON content
with open("/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/clean_filetree/summarized_filetree.json", "r") as json_file:
    content_json = json.load(json_file)

# Load the text file content
with open("/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/clean_filetree/file_list.txt", "r") as txt_file:
    file_list = txt_file.read()

# Run the functions to restructure and format the filetree
new_filetree = structure_filetree(content_json, file_list)
formatted_filetree = format_filetree_text(new_filetree)

# Save the output as a new JSON file
output_path = "/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/structure_filetree/filetree.json"
with open(output_path, "w") as output_json:
    json.dump(formatted_filetree, output_json, indent=4)

print(f"Filetree has been processed and saved to {output_path}")
