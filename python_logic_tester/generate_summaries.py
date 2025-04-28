import sys
import os
import json

from openai import OpenAI
client = OpenAI(api_key="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")

def summarize_functions(python_file_string):
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You are a python code summarizer. You will see python code. You should summarize all functions based on what they do with respect to the whole script.",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "import random\n\ndef generate_random_numbers(n, lower, upper):\n    return [random.randint(lower, upper) for _ in range(n)]\n\ndef calculate_mean(numbers):\n    return sum(numbers) / len(numbers)\n\ndef calculate_median(numbers):\n    sorted_numbers = sorted(numbers)\n    mid = len(sorted_numbers) // 2\n    return (sorted_numbers[mid] if len(sorted_numbers) % 2 != 0 \n            else (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2)\n\ndef calculate_variance(numbers, mean):\n    return sum((x - mean) ** 2 for x in numbers) / len(numbers)\n\ndef calculate_standard_deviation(variance):\n    return variance ** 0.5\n\ndef sort_numbers(numbers, descending=False):\n    return sorted(numbers, reverse=descending)\n\ndef generate_statistics(numbers):\n    mean = calculate_mean(numbers)\n    median = calculate_median(numbers)\n    variance = calculate_variance(numbers, mean)\n    std_deviation = calculate_standard_deviation(variance)\n    return {\n        'mean': mean,\n        'median': median,\n        'variance': variance,\n        'std_deviation': std_deviation\n    }\n\ndef print_statistics(stats):\n    for key, value in stats.items():\n        print(f\"{key.capitalize()}: {value:.2f}\")\n\ndef main():\n    n = 50\n    lower_bound = 10\n    upper_bound = 100\n    random_numbers = generate_random_numbers(n, lower_bound, upper_bound)\n    sorted_numbers = sort_numbers(random_numbers)\n    stats = generate_statistics(sorted_numbers)\n    print_statistics(stats)\n\nmain()"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "{\"function_information\":[{\"function_name\":\"generate_random_numbers\",\"function_description\":\"Generates a list of 'n' random integers between 'lower' and 'upper' bounds.\"},{\"function_name\":\"calculate_mean\",\"function_description\":\"Calculates and returns the mean (average) of a list of numbers.\"},{\"function_name\":\"calculate_median\",\"function_description\":\"Calculates and returns the median of a list of numbers.\"},{\"function_name\":\"calculate_variance\",\"function_description\":\"Calculates and returns the variance of a list of numbers based on the given mean.\"},{\"function_name\":\"calculate_standard_deviation\",\"function_description\":\"Calculates and returns the standard deviation from the variance.\"},{\"function_name\":\"sort_numbers\",\"function_description\":\"Sorts the list of numbers in ascending or descending order based on the 'descending' parameter.\"},{\"function_name\":\"generate_statistics\",\"function_description\":\"Generates and returns a dictionary containing the mean, median, variance, and standard deviation of a list of numbers.\"},{\"function_name\":\"print_statistics\",\"function_description\":\"Prints the statistical metrics (mean, median, variance, standard deviation) in a formatted manner.\"},{\"function_name\":\"main\",\"function_description\":\"Main function that ties together the generation of random numbers, sorting, calculating statistics, and printing the results.\"}]}"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": python_file_string
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
        "name": "function_descriptions",
        "schema": {
            "type": "object",
            "required": [
            "function_information"
            ],
            "properties": {
            "function_information": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "function_name",
                    "function_description"
                ],
                "properties": {
                    "function_name": {
                    "type": "string"
                    },
                    "function_description": {
                    "type": "string"
                    }
                },
                "additionalProperties": False
                }
            }
            },
            "additionalProperties": False
        },
        "strict": True
        }
    }
    )
    text = response.choices[0].message.content
    return text

def summarize_functions_in_directory(directory_path, output_json_path):
    function_summaries = {}

    # Walk through all files and subdirectories within the directory
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.py'):
                # Get the full file path
                file_path = os.path.join(root, filename)
                
                # Compute the relative path from the root directory
                relative_path = os.path.relpath(file_path, directory_path)
                
                with open(file_path, 'r') as file:
                    python_file_string = file.read()
                
                # Get the function summaries using the summarize_functions function
                try:
                    summary_json = summarize_functions(python_file_string)
                    # Parse the JSON string into a Python dictionary
                    summary = json.loads(summary_json)
                except Exception as e:
                    print(f"Error summarizing {relative_path}: {e}")
                    summary = {"error": str(e)}
                
                # Add the summary to the dictionary with the relative path as the key
                function_summaries[relative_path] = summary

    # Write the collected summaries to the output JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(function_summaries, json_file, indent=4)

    print(f"Function summaries have been written to {output_json_path}")



# Example usage
if __name__ == "__main__":
    directory_path = '/Users/typham-swann/Desktop/research_code_app/python_logic_tester/sample_directory'  # Replace with your directory path
    output_json_path = '/Users/typham-swann/Desktop/research_code_app/python_logic_tester/function_descriptions.json'    # Output JSON file path

    summarize_functions_in_directory(directory_path, output_json_path)
