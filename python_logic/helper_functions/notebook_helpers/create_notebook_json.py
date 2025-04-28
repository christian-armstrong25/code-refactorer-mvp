from openai import OpenAI
import json
client = OpenAI(api_key="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")

def ai_helper_create_notebook_object(notebook_string):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are in charge of formatting a python notebook. You will receive a string containing the contents of a notebook where the code and markdown are specified. Please convert it into the format for pynb. "
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Notebook String: ```{notebook_string}```"
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
        "name": "ipynb_notebook",
        "schema": {
            "type": "object",
            "required": [
            "metadata",
            "nbformat",
            "nbformat_minor",
            "cells"
            ],
            "properties": {
            "cells": {
                "type": "array",
                "items": {
                "type": "object",
                "required": [
                    "cell_type",
                    "source",
                    "execution_count",
                    "outputs"
                ],
                "properties": {
                    "source": {
                    "type": "string"
                    },
                    "outputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                        "output_type",
                        "text",
                        "name",
                        "execution_count",
                        "ename",
                        "evalue",
                        "traceback"
                        ],
                        "properties": {
                        "data": {
                            "type": "object",
                            "additionalProperties": False
                        },
                        "name": {
                            "type": "string"
                        },
                        "text": {
                            "type": "string"
                        },
                        "ename": {
                            "type": "string"
                        },
                        "evalue": {
                            "type": "string"
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": False
                        },
                        "traceback": {
                            "type": "array",
                            "items": {
                            "type": "string"
                            }
                        },
                        "output_type": {
                            "type": "string"
                        },
                        "execution_count": {
                            "type": [
                            "integer",
                            "null"
                            ]
                        }
                        },
                        "additionalProperties": False
                    }
                    },
                    "metadata": {
                    "type": "object",
                    "additionalProperties": False
                    },
                    "cell_type": {
                    "enum": [
                        "code",
                        "markdown"
                    ],
                    "type": "string"
                    },
                    "execution_count": {
                    "type": [
                        "integer",
                        "null"
                    ]
                    }
                },
                "additionalProperties": False
                }
            },
            "metadata": {
                "type": "object",
                "required": [
                "kernel_info",
                "language_info"
                ],
                "properties": {
                "kernel_info": {
                    "type": "object",
                    "required": [
                    "name"
                    ],
                    "properties": {
                    "name": {
                        "type": "string"
                    }
                    },
                    "additionalProperties": False
                },
                "language_info": {
                    "type": "object",
                    "required": [
                    "name",
                    "version",
                    "codemirror_mode"
                    ],
                    "properties": {
                    "name": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string"
                    },
                    "codemirror_mode": {
                        "type": "string"
                    }
                    },
                    "additionalProperties": False
                }
                },
                "additionalProperties": False
            },
            "nbformat": {
                "type": "integer"
            },
            "nbformat_minor": {
                "type": "integer"
            }
            },
            "additionalProperties": False
        },
        "strict": True
        }
    }
    )
    notebook_object = json.loads(response.choices[0].message.content)
    return notebook_object

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
    
    
# Load the notebook plan from a txt file
notebook_plan_filepath = "/Users/typham-swann/Desktop/research_code_app/python_logic/helper_functions/notebook_helpers/test_outputs/notebook_string.txt"  # Replace this path with the actual file path
notebook_plan = load_text(notebook_plan_filepath)

notebook_output = ai_helper_create_notebook_object(notebook_plan)

# Save the result to a new JSON file
output_filepath = "/Users/typham-swann/Desktop/research_code_app/python_logic/helper_functions/notebook_helpers/test_outputs/notebook_output.json"
save_json(output_filepath, notebook_output)