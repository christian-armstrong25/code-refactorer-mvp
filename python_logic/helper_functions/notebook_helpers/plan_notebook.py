from openai import OpenAI
import json
client = OpenAI(api_key="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")


def ai_helper_plan_notebook(content_object, filetree_object):
    content_object = helper_filter_code_fields(content_object)

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are tasked with planning a turnkey Jupiter notebook for a codebase directory. You must understand the project and create a comprehensive plan for what an ideal notebook would look like for someone who has never before seen the project but would like to get started. You will be provided a filetree and a description of the python files within the project. Your plan should mention all necessary code and functions that must be included in the notebook, including mentioning which files the functions should come from. You do not need to provide the actual code, as the creator of the notebook will have the project code; simply, you need to say which functions should be used and from which files."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Filetree Object: ```{\n    \"name\": \"nlp_project\",\n    \"children\": [\n        {\n            \"name\": \"data\",\n            \"children\": [\n                {\n                    \"name\": \"raw\",\n                    \"children\": [\n                        {\n                            \"name\": \"reviews.csv\",\n                            \"children\": [],\n                            \"is_directory\": false\n                        }\n                    ],\n                    \"is_directory\": true\n                },\n                {\n                    \"name\": \"processed\",\n                    \"children\": [\n                        {\n                            \"name\": \"cleaned_reviews.csv\",\n                            \"children\": [],\n                            \"is_directory\": false\n                        }\n                    ],\n                    \"is_directory\": true\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"models\",\n            \"children\": [\n                {\n                    \"name\": \"sentiment_model.pkl\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"scripts\",\n            \"children\": [\n                {\n                    \"name\": \"preprocessing.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"modeling.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"visualization.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"data_loading.py\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"docs\",\n            \"children\": [\n                {\n                    \"name\": \"project_overview.md\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"function_descriptions.json\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        },\n        {\n            \"name\": \"output\",\n            \"children\": [\n                {\n                    \"name\": \"evaluation_report.txt\",\n                    \"children\": [],\n                    \"is_directory\": false\n                },\n                {\n                    \"name\": \"word_cloud.png\",\n                    \"children\": [],\n                    \"is_directory\": false\n                }\n            ],\n            \"is_directory\": true\n        }\n    ],\n    \"is_directory\": true\n}```\n\nPython Content: ```{\n    \"nlp_project/preprocessing.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"tokenize_text\",\n                \"function_description\": \"Splits the input text into tokens (words) using the nltk library and returns a list of tokens.\"\n            },\n            {\n                \"function_name\": \"remove_stopwords\",\n                \"function_description\": \"Removes common English stopwords from the token list using a predefined stopword list from the nltk library.\"\n            },\n            {\n                \"function_name\": \"stem_tokens\",\n                \"function_description\": \"Applies stemming to the token list using the PorterStemmer from the nltk library, returning the stemmed version of each token.\"\n            }\n        ],\n        \"file_summary\": \"The `preprocessing.py` script handles text preprocessing tasks essential for preparing text data for sentiment analysis. It leverages the nltk library for natural language processing functions. The script defines three main functions:\\n\\n1. `tokenize_text(text)`: This function takes an input string and splits it into tokens (words), returning the list of tokens for further processing.\\n\\n2. `remove_stopwords(tokens)`: Removes common English stopwords from the token list to retain only the most meaningful words. It uses a stopword list from nltk for this purpose.\\n\\n3. `stem_tokens(tokens)`: Applies stemming to reduce words to their base or root forms using the PorterStemmer from nltk. This is useful for standardizing words and improving model performance.\\n\\nThe preprocessing steps in this script are crucial for cleaning and transforming text data into a suitable format for sentiment analysis models, ensuring that only relevant information is passed on for further modeling steps.\"\n    },\n    \"nlp_project/modeling.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"train_sentiment_model\",\n                \"function_description\": \"Trains a sentiment analysis model using a TF-IDF vectorizer and a Logistic Regression classifier on the provided training data.\"\n            },\n            {\n                \"function_name\": \"predict_sentiment\",\n                \"function_description\": \"Generates sentiment predictions (positive, negative, or neutral) for input text data using the trained sentiment analysis model.\"\n            },\n            {\n                \"function_name\": \"evaluate_model\",\n                \"function_description\": \"Evaluates the trained sentiment analysis model using test data and calculates accuracy, precision, and recall scores.\"\n            }\n        ],\n        \"file_summary\": \"The `modeling.py` script is dedicated to building and evaluating a sentiment analysis model. It utilizes scikit-learn's Logistic Regression classifier and TF-IDF vectorizer for text feature extraction and model training. The script includes three main functions:\\n\\n1. `train_sentiment_model(X_train, y_train)`: This function takes input features (`X_train`) and target labels (`y_train`) and trains a sentiment analysis model using a TF-IDF vectorizer to transform text data into numerical form. It returns the trained model.\\n\\n2. `predict_sentiment(model, X_test)`: Uses the trained model to predict the sentiment of new input data (`X_test`), returning the predictions as an array of sentiment labels (e.g., positive, negative, or neutral).\\n\\n3. `evaluate_model(model, X_test, y_test)`: Evaluates the model’s performance using the test set, `X_test` and `y_test`, calculating key metrics like accuracy, precision, and recall scores.\\n\\nThe script is designed to support the full modeling pipeline, from training to evaluation, and works with preprocessed text data. It plays a crucial role in developing and fine-tuning the sentiment analysis model.\"\n    },\n    \"nlp_project/visualization.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"plot_sentiment_distribution\",\n                \"function_description\": \"Plots a bar chart showing the distribution of sentiments (positive, negative, neutral) in the dataset using Matplotlib.\"\n            },\n            {\n                \"function_name\": \"plot_word_cloud\",\n                \"function_description\": \"Generates and displays a word cloud visualization of the most frequent words in the text data using the WordCloud library.\"\n            }\n        ],\n        \"file_summary\": \"The `visualization.py` script contains functions for visualizing text data and model outputs. It includes two main functions:\\n\\n1. `plot_sentiment_distribution(data)`: This function takes a dataset containing sentiment labels (positive, negative, neutral) and plots a bar chart showing the distribution of these labels using Matplotlib.\\n\\n2. `plot_word_cloud(text)`: Generates a word cloud visualization from the input text data, highlighting the most frequently occurring words. It uses the WordCloud library for this task.\\n\\nThese visualizations are essential for understanding the characteristics of the dataset and the outputs of the sentiment analysis model. The script is designed to provide quick, informative visual feedback during the development and evaluation of the sentiment analysis pipeline.\"\n    },\n    \"nlp_project/data_loading.py\": {\n        \"function_information\": [\n            {\n                \"function_name\": \"load_dataset\",\n                \"function_description\": \"Loads the text dataset from a CSV file given by 'filepath' into a pandas DataFrame.\"\n            },\n            {\n                \"function_name\": \"split_data\",\n                \"function_description\": \"Splits the dataset into training and test sets, returning the features and labels for each split.\"\n            },\n            {\n                \"function_name\": \"save_processed_data\",\n                \"function_description\": \"Saves the processed dataset to a CSV file at the specified 'filepath' without the index.\"\n            }\n        ],\n        \"file_summary\": \"The `data_loading.py` script manages the loading, splitting, and saving of datasets used in the NLP project. It uses pandas for handling CSV data and defines three main functions:\\n\\n1. `load_dataset(filepath)`: This function reads a CSV file specified by `filepath` into a pandas DataFrame, which serves as the input for the NLP pipeline.\\n\\n2. `split_data(data)`: Splits the DataFrame into training and test sets, returning the features (text) and labels (sentiments) for each set. This is essential for training and evaluating the sentiment analysis model.\\n\\n3. `save_processed_data(data, filepath)`: Saves the processed DataFrame to a CSV file at the specified `filepath`, excluding the index to ensure compatibility with other scripts in the pipeline.\\n\\nThe script serves as the foundation for data management in the NLP project, ensuring that text data is properly loaded, split, and saved for subsequent processing and modeling tasks.\"\n    }\n}\n```"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "Here's a detailed plan for creating a comprehensive Jupyter notebook for the current NLP project. The goal is to guide a new user through the entire workflow, from data loading and preprocessing to modeling, evaluation, and visualization.\n\n### Notebook Structure and Content\n\n1. **Notebook Setup**\n   - **Introduction**\n     - Brief overview of the project, explaining the goal (e.g., performing sentiment analysis on text data) and the steps involved.\n   - **Environment Setup**\n     - Import necessary libraries (e.g., pandas, nltk, sklearn, matplotlib, etc.).\n\n2. **Data Loading**\n   - **Load Dataset**\n     - Use `load_dataset(filepath)` from `data_loading.py` to load the raw data from `data/raw/reviews.csv` into a pandas DataFrame.\n     - Display the first few rows of the dataset to understand its structure.\n\n3. **Data Preprocessing**\n   - **Text Tokenization**\n     - Use `tokenize_text(text)` from `preprocessing.py` to split input text into tokens.\n   - **Stopwords Removal**\n     - Use `remove_stopwords(tokens)` from `preprocessing.py` to filter out common stopwords and prepare the text for analysis.\n   - **Stemming**\n     - Use `stem_tokens(tokens)` from `preprocessing.py` to perform stemming and reduce words to their root form.\n\n4. **Data Splitting**\n   - **Training and Test Sets**\n     - Utilize `split_data(data)` from `data_loading.py` to divide the preprocessed data into training and test sets.\n     - Display the dimensions of the resulting datasets to verify the split.\n\n5. **Model Training**\n   - **Train Model**\n     - Use `train_sentiment_model(X_train, y_train)` from `modeling.py` to train a sentiment analysis model on the training data.\n     - Provide an overview of the model's basic configuration and feature extraction method (e.g., TF-IDF).\n\n6. **Model Evaluation**\n   - **Predict Sentiments**\n     - Use `predict_sentiment(model, X_test)` from `modeling.py` to generate sentiment predictions on the test set.\n   - **Evaluate Model**\n     - Use `evaluate_model(model, X_test, y_test)` from `modeling.py` to calculate and display model performance metrics such as accuracy, precision, and recall.\n\n7. **Results Visualization**\n   - **Sentiment Distribution**\n     - Use `plot_sentiment_distribution(data)` from `visualization.py` to plot a bar chart of sentiment distribution in the dataset.\n   - **Word Cloud Visualization**\n     - Use `plot_word_cloud(text)` from `visualization.py` to generate word clouds for visual representation of the most frequent words.\n\n8. **Saving Processed Data**\n   - **Save Data**\n     - Use `save_processed_data(data, filepath)` from `data_loading.py` to save the processed dataset to `data/processed/cleaned_reviews.csv`.\n\n9. **Conclusion and Next Steps**\n   - Summarize key findings, insights gained, and possible further exploration.\n   - Provide suggestions for extending the project or improving model performance, such as experimenting with different algorithms or preprocessing techniques.\n\n---\n\nThis structure ensures a logical flow that guides the user through understanding and running the project's essential components while keeping the notebook organized and educational. Each step should include appropriate comments, explanations, and visualization outputs where applicable to enhance understanding and engagement."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Filetree Object: ```{filetree_object}```\n\nPython Content: ```{content_object}```"
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
    notebook_plan = response.choices[0].message.content
    return notebook_plan

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


# FOR TESTING BELOW

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def save_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)
    
content_object = load_json("/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/fix_differences/python_content.json")
    
filetree_json = "/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/structure_filetree/filetree.json"    

notebook_output = ai_helper_plan_notebook(content_object, filetree_json)

# Save the result to a new txt file
output_filepath = "/Users/typham-swann/Desktop/research_code_app/python_logic/helper_functions/notebook_helpers/test_outputs/notebook_output.txt"
with open(output_filepath, 'w') as file:
    file.write(notebook_output)
