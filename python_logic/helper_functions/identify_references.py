from openai import OpenAI
import json
client = OpenAI(api_key="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")

def identify_references(python_text):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are tasked with identifying filepaths within a python file. You will see python code from a python file. You must respond with all external filepaths. It's important that you included all external filepaths identified. Be sure to include external and internal files, using the exact path used in the code."
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

# example usage
sample_code = "import os\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n# Importing utility functions from other modules\nfrom utils.data_loader import load_csv_data\nfrom utils.preprocessing import preprocess_data, filter_outliers\nfrom utils.statistics import calculate_summary_statistics, calculate_correlations\n\n# Main analysis function\ndef analyze_data():\n    # Step 1: Define the path to the data\n    base_dir = os.path.dirname(os.path.abspath(__file__))\n    data_path = os.path.join(base_dir, '../data/data.csv')\n    \n    # Step 2: Load the data using the utility function\n    data = load_csv_data(data_path)\n    if data is None:\n        return\n    \n    # Step 3: Preprocess the data (cleaning and outlier removal)\n    data = preprocess_data(data)\n    data = filter_outliers(data, 'column_of_interest', threshold=3)  # Example outlier filtering\n\n    # Step 4: Calculate summary statistics\n    summary_stats = calculate_summary_statistics(data)\n    print(\"Summary Statistics:\")\n    print(summary_stats)\n\n    # Step 5: Calculate correlations between columns\n    correlations = calculate_correlations(data)\n    print(\"Correlations:\")\n    print(correlations)\n\n    # Step 6: Plot histogram of the cleaned data\n    plt.figure(figsize=(10, 6))\n    plt.hist(data['column_of_interest'], bins=30, alpha=0.7)\n    plt.title('Distribution of Column of Interest')\n    plt.xlabel('Value')\n    plt.ylabel('Frequency')\n    plt.show()\n\nif __name__ == '__main__':\n    analyze_data()\n"
print(identify_references(sample_code))