import json
from openai import OpenAI
client = OpenAI(api_key="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")


def summarize_files(python_content_obj):
    summarized_content = {}

    for filepath, file_info in python_content_obj.items():
        # Extract relevant information for the summary
        code = file_info.get("code", "")
        function_information = file_info.get("function_information", [])

        # Generate the file summary using the ai_summarize_files function
        file_summary = ai_summarize_files(
            filetree_obj=python_content_obj,  # Assuming this is passed as the full python content
            project_function_information=function_information,
            python_text=code,
            python_filepath=filepath
        )

        # Copy existing file info and add the new summary
        summarized_content[filepath] = {
            "code": code,
            "function_information": function_information,
            "file_summary": file_summary
        }

    return summarized_content

def ai_summarize_files(filetree_obj, project_function_information, python_text, python_filepath):
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You will receive a filetree in JSON format, a list of function descriptions for the whole project in JSON form, and a python file. Please create a short summary of the what the python file does. Include all instances of outside files or functions being referenced or necessary in the python file. Please respond with only the file information.",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Filetree JSON: ```{\n  \"name\": \"DataAnalysisProject\",\n  \"children\": [\n    {\n      \"name\": \"data\",\n      \"children\": [\n        {\n          \"name\": \"dataset.csv\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"scripts\",\n      \"children\": [\n        {\n          \"name\": \"data_processing.py\",\n          \"children\": [],\n          \"is_directory\": false\n        },\n        {\n          \"name\": \"visualization.py\",\n          \"children\": [],\n          \"is_directory\": false\n        },\n        {\n          \"name\": \"model_training.py\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"output\",\n      \"children\": [\n        {\n          \"name\": \"results.txt\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"README.md\",\n      \"children\": [],\n      \"is_directory\": false\n    }\n  ],\n  \"is_directory\": true\n}```\n\nProject Function information: ```{\n    \"scripts/model_training.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"train_model\",\n                \"function_description\": \"Trains a linear regression model using input features 'X' and target values 'y'. Returns the trained model.\"\n            },\n            {\n                \"function_name\": \"predict\",\n                \"function_description\": \"Generates predictions for new input data 'X_new' using the provided trained model.\"\n            },\n            {\n                \"function_name\": \"evaluate_model\",\n                \"function_description\": \"Evaluates the performance of the trained model on test data 'X_test' and 'y_test', returning the model's score.\"\n            }\n        ]\n    },\n    \"scripts/visualization.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"plot_data\",\n                \"function_description\": \"Plots a graph of the provided data, using 'date' for the x-axis and 'value' for the y-axis, and displays the plot.\"\n            },\n            {\n                \"function_name\": \"save_plot\",\n                \"function_description\": \"Plots a graph of the provided data, using 'date' for the x-axis and 'value' for the y-axis, and saves the plot to the specified file path.\"\n            }\n        ]\n    },\n    \"scripts/data_processing.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"load_data\",\n                \"function_description\": \"Loads data from a CSV file specified by the 'filepath' and returns it as a pandas DataFrame.\"\n            },\n            {\n                \"function_name\": \"clean_data\",\n                \"function_description\": \"Cleans the data by removing rows with missing values and filtering out rows where the 'value' column is negative.\"\n            },\n            {\n                \"function_name\": \"save_clean_data\",\n                \"function_description\": \"Saves the cleaned DataFrame to a specified CSV file without including the index column.\"\n            }\n        ]\n    }\n}```\n\nscripts/model_training.py: ```from sklearn.linear_model import LinearRegression\n\ndef train_model(X, y):\n    model = LinearRegression()\n    model.fit(X, y)\n    return model\n\ndef predict(model, X_new):\n    predictions = model.predict(X_new)\n    return predictions\n\ndef evaluate_model(model, X_test, y_test):\n    score = model.score(X_test, y_test)\n    return score\n``` "
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "scripts/model_training.py is responsible for training, predicting, and evaluating a linear regression model, forming the core of the project's machine learning component. It interacts with processed data (likely prepared by data_processing.py) and its output can be visualized using functions in visualization.py. It expects cleaned and structured data as input, indicating its role after data preprocessing and before results visualization in the overall project workflow."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Filetree JSON: ```{\n  \"name\": \"UrbanDevelopmentStudy\",\n  \"children\": [\n    {\n      \"name\": \"data\",\n      \"children\": [\n        {\n          \"name\": \"population_data\",\n          \"children\": [\n            {\n              \"name\": \"city_population.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"urban_growth.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"economic_data\",\n          \"children\": [\n            {\n              \"name\": \"gdp_per_capita.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"employment_rate.csv\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"scripts\",\n      \"children\": [\n        {\n          \"name\": \"data_preprocessing\",\n          \"children\": [\n            {\n              \"name\": \"load_data.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"clean_data.py\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"analysis\",\n          \"children\": [\n            {\n              \"name\": \"population_trends.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"economic_analysis.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"urban_growth_analysis.py\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"visualization\",\n          \"children\": [\n            {\n              \"name\": \"plot_population.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"plot_economic.py\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"plot_urban_growth.py\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"results\",\n      \"children\": [\n        {\n          \"name\": \"summary_statistics\",\n          \"children\": [\n            {\n              \"name\": \"population_summary.txt\",\n              \"children\": [],\n              \"is_directory\": false\n            },\n            {\n              \"name\": \"economic_summary.txt\",\n              \"children\": [],\n              \"is_directory\": false\n            }\n          ],\n          \"is_directory\": true\n        },\n        {\n          \"name\": \"final_report.pdf\",\n          \"children\": [],\n          \"is_directory\": false\n        }\n      ],\n      \"is_directory\": true\n    },\n    {\n      \"name\": \"README.md\",\n      \"children\": [],\n      \"is_directory\": false\n    }\n  ],\n  \"is_directory\": true\n}```\n\nProject Function Information: ```{\n  \"scripts/data_preprocessing/load_data.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"load_population_data\",\n        \"function_description\": \"Loads population data from CSV files and returns them as a dictionary of pandas DataFrames.\"\n      },\n      {\n        \"function_name\": \"load_economic_data\",\n        \"function_description\": \"Loads economic data from CSV files and returns them as a dictionary of pandas DataFrames.\"\n      }\n    ]\n  },\n  \"scripts/data_preprocessing/clean_data.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"remove_missing_values\",\n        \"function_description\": \"Removes rows with missing values from all DataFrames in the input dictionary and returns the cleaned dictionary.\"\n      },\n      {\n        \"function_name\": \"filter_data_by_year\",\n        \"function_description\": \"Filters all DataFrames in the input dictionary to include data from a specified range of years and returns the filtered dictionary.\"\n      }\n    ]\n  },\n  \"scripts/analysis/population_trends.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"calculate_population_growth_rate\",\n        \"function_description\": \"Calculates the annual growth rate of the population for each city and returns the resulting DataFrame.\"\n      },\n      {\n        \"function_name\": \"identify_rapid_growth_cities\",\n        \"function_description\": \"Identifies cities with a population growth rate above a specified threshold and returns a list of these cities.\"\n      }\n    ]\n  },\n  \"scripts/analysis/economic_analysis.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"calculate_gdp_growth\",\n        \"function_description\": \"Calculates the annual GDP growth rate for each city and returns the resulting DataFrame.\"\n      },\n      {\n        \"function_name\": \"calculate_employment_rate_change\",\n        \"function_description\": \"Calculates the change in employment rate over time for each city and returns the resulting DataFrame.\"\n      }\n    ]\n  },\n  \"scripts/analysis/urban_growth_analysis.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"calculate_urban_growth_rate\",\n        \"function_description\": \"Calculates the annual urban growth rate for each city and returns the resulting DataFrame.\"\n      },\n      {\n        \"function_name\": \"correlate_population_and_urban_growth\",\n        \"function_description\": \"Calculates the correlation between population growth and urban area expansion for each city and returns a correlation matrix.\"\n      }\n    ]\n  },\n  \"scripts/visualization/plot_population.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"plot_population_trend\",\n        \"function_description\": \"Plots the population trend over time for a given city and displays the plot.\"\n      },\n      {\n        \"function_name\": \"plot_population_comparison\",\n        \"function_description\": \"Plots a comparison of population growth rates between multiple cities and displays the plot.\"\n      }\n    ]\n  },\n  \"scripts/visualization/plot_economic.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"plot_gdp_trend\",\n        \"function_description\": \"Plots the GDP trend over time for a given city and displays the plot.\"\n      },\n      {\n        \"function_name\": \"plot_employment_rate\",\n        \"function_description\": \"Plots the employment rate change over time for a given city and displays the plot.\"\n      }\n    ]\n  },\n  \"scripts/visualization/plot_urban_growth.py\": {\n    \"function_information\": [\n      {\n        \"function_name\": \"plot_urban_expansion\",\n        \"function_description\": \"Plots the urban expansion trend over time for a given city and displays the plot.\"\n      },\n      {\n        \"function_name\": \"plot_population_vs_urban_growth\",\n        \"function_description\": \"Plots the relationship between population growth and urban area expansion for a given city and displays the plot.\"\n      }\n    ]\n  }\n}```\n\ndata_preprocessing/clean_data.py: ```def remove_missing_values(data_dict):\n    cleaned_data = {key: df.dropna() for key, df in data_dict.items()}\n    return cleaned_data\n\ndef filter_data_by_year(data_dict, start_year, end_year):\n    filtered_data = {key: df[(df['year'] >= start_year) & (df['year'] <= end_year)] for key, df in data_dict.items()}\n    return filtered_data\n```"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "The `clean_data.py` script is used for preprocessing datasets related to urban development, specifically focusing on handling missing data and time-specific data filtering. It contains two main functions: remove_missing_values(data_dict)` which takes a dictionary of pandas DataFrames (likely provided by `load_data.py`), iterates over each DataFrame, removes rows with any missing values, and returns a cleaned dictionary of DataFrames without missing data.\n`filter_data_by_year(data_dict, start_year, end_year)` filters the data in each DataFrame within the input dictionary to include only the rows with 'year' values between the specified start and end years, returning the filtered dictionary of DataFrames.\n\nThe script ensures that the data is clean and relevant to the specified time period before it's further analyzed or visualized in other parts of the project."
            }
        ]
        },
                {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Filetree JSON: ```{filetree_obj}```\n\nProject Function information: ```{project_function_information}```\n\n{python_filepath}: ```{python_text}``` "
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
        "type": "text"
    }
    )
    text = response.choices[0].message.content
    return text

# Example usage:
with open('/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/summarize_functions/python_content.json', 'r') as file:
    data = json.load(file)

# Call summarize_files and store the new JSON with summaries
summarized_python_content = summarize_files(data)

# Output the new JSON object with summaries to a file (if needed)
with open('/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/summarize_file/python_content.json', 'w') as json_file:
    json.dump(summarized_python_content, json_file, indent=4)
