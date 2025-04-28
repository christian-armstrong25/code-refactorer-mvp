from openai import OpenAI
import re
import json
client = OpenAI(api_key="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")


def comment_code(python_files_content):
    """
    Loops through the dictionary, applies the `comment_code` function to each 
    Python code content string, and returns a new dictionary with the modified content.
    
    :param python_files_content: Dictionary with file paths as keys and Python code as values
    :return: New dictionary with the commented Python code
    """
    # Dictionary to store the updated content
    updated_python_files_content = {}

    # Iterate through the dictionary
    for file_path, file_content in python_files_content.items():
        # Apply the comment_code function to the file content
        updated_content = ai_comment(file_content)
        # Store the updated content in the new dictionary
        updated_python_files_content[file_path] = updated_content

    return updated_python_files_content


def ai_comment(python_file_string):
    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "text": "You are a code commenter. You will see python code. You should reply with a commented version of the code. Please only include one single comment per function. It is important that you don't change the code. It is important that you respond with the entirety of the code. ",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "def generate_squares(n):\n    return [x**2 for x in range(1, n + 1)]\n\ndef filter_even_squares(squares):\n    return [square for square in squares if square % 2 == 0]\n\ndef main():\n    n = 20\n    squares = generate_squares(n)\n    even_squares = filter_even_squares(squares)\n    print(f\"Squares from 1 to {n}: {squares}\")\n    print(f\"Even squares: {even_squares}\")\n\nmain()"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "# Function to generate a list of squares for numbers from 1 to n\ndef generate_squares(n):\n    return [x**2 for x in range(1, n + 1)]\n\n# Function to filter out only even squares from a list of squares\ndef filter_even_squares(squares):\n    return [square for square in squares if square % 2 == 0]\n\n# Main function to run the program\ndef main():\n    n = 20\n    squares = generate_squares(n) \n    even_squares = filter_even_squares(squares)\n    print(f\"Squares from 1 to {n}: {squares}\")\n    print(f\"Even squares: {even_squares}\")\n\nmain()"
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
            "text": "import random\n\n# Generates a list of n random integers between lower and upper bounds\ndef generate_random_numbers(n, lower, upper):\n    return [random.randint(lower, upper) for _ in range(n)]\n\n# Calculates the mean of a list of numbers\ndef calculate_mean(numbers):\n    return sum(numbers) / len(numbers)\n\n# Calculates the median of a list of numbers\ndef calculate_median(numbers):\n    sorted_numbers = sorted(numbers)\n    mid = len(sorted_numbers) // 2\n    return (sorted_numbers[mid] if len(sorted_numbers) % 2 != 0 \n            else (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2)\n\n# Calculates the variance of a list of numbers given the mean\ndef calculate_variance(numbers, mean):\n    return sum((x - mean) ** 2 for x in numbers) / len(numbers)\n\n# Calculates the standard deviation given the variance\ndef calculate_standard_deviation(variance):\n    return variance ** 0.5\n\n# Sorts a list of numbers in ascending or descending order\ndef sort_numbers(numbers, descending=False):\n    return sorted(numbers, reverse=descending)\n\n# Generates a dictionary of statistical measures for a list of numbers\ndef generate_statistics(numbers):\n    mean = calculate_mean(numbers)\n    median = calculate_median(numbers)\n    variance = calculate_variance(numbers, mean)\n    std_deviation = calculate_standard_deviation(variance)\n    return {\n        'mean': mean,\n        'median': median,\n        'variance': variance,\n        'std_deviation': std_deviation\n    }\n\n# Prints the statistical measures in a formatted manner\ndef print_statistics(stats):\n    for key, value in stats.items():\n        print(f\"{key.capitalize()}: {value:.2f}\")\n\n# Main function to generate random numbers and display their statistics\ndef main():\n    n = 50\n    lower_bound = 10\n    upper_bound = 100\n    random_numbers = generate_random_numbers(n, lower_bound, upper_bound)\n    sorted_numbers = sort_numbers(random_numbers)\n    stats = generate_statistics(sorted_numbers)\n    print_statistics(stats)\n\nmain()"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\nclass SimpleNN(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super(SimpleNN, self).__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(hidden_size, output_size)\n\n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        return x\n\ninput_size = 10\nhidden_size = 20\noutput_size = 1\nlearning_rate = 0.001\nepochs = 100\n\nmodel = SimpleNN(input_size, hidden_size, output_size)\ncriterion = nn.MSELoss()\noptimizer = optim.Adam(model.parameters(), lr=learning_rate)\n\nx_train = torch.randn(100, input_size)\ny_train = torch.randn(100, output_size)\n\ndataset = TensorDataset(x_train, y_train)\ndataloader = DataLoader(dataset, batch_size=10, shuffle=True)\n\nfor epoch in range(epochs):\n    for inputs, targets in dataloader:\n        outputs = model(inputs)\n        loss = criterion(outputs, targets)\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "import torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import DataLoader, TensorDataset\n\n# Simple Neural Network definition with one hidden layer\nclass SimpleNN(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super(SimpleNN, self).__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(hidden_size, output_size)\n\n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        return x\n\ninput_size = 10\nhidden_size = 20\noutput_size = 1\nlearning_rate = 0.001\nepochs = 100\n\nmodel = SimpleNN(input_size, hidden_size, output_size)\ncriterion = nn.MSELoss()\noptimizer = optim.Adam(model.parameters(), lr=learning_rate)\n\nx_train = torch.randn(100, input_size)\ny_train = torch.randn(100, output_size)\n\ndataset = TensorDataset(x_train, y_train)\ndataloader = DataLoader(dataset, batch_size=10, shuffle=True)\n\nfor epoch in range(epochs):\n    for inputs, targets in dataloader:\n        outputs = model(inputs)\n        loss = criterion(outputs, targets)\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()"
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
    temperature=0,
    max_tokens=16380,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "text"
    }
    )

    text = response.choices[0].message.content

    if text.startswith('```python\n') and text.endswith('```'):
        # Remove the wrapping backticks and the language identifier
        return text[10:-3].strip()
    
    return text
    


# Example usage (assuming you already have the comment_code function and python_files_content dictionary)
input_file_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/handle_files/python_content.json'
with open(input_file_path, 'r') as json_file:
    python_files_content = json.load(json_file)

# Apply the comment_code function to the loaded content
commented_files_content = comment_code(python_files_content)

# Output the updated dictionary to verify
output_file_path = '/Users/typham-swann/Desktop/research_code_app/python_logic/final_pipeline_functions/test_outputs/comment_code/commented_python_content.json'
with open(output_file_path, 'w') as json_file:
    json.dump(commented_files_content, json_file, indent=4)

print(f"Updated Python files content has been written to {output_file_path}")