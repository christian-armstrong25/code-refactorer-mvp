from helpers.query import query_model
import ast
import re

def read_python_file(file_path: str) -> str:  
    """  
    Reads the content of a Python file and returns it as a string.  
      
    :param file_path: The path to the Python file.  
    :return: The content of the file as a string.  
    """  
    try:  
        with open(file_path, 'r', encoding='utf-8') as file:  
            file_content = file.read()  
    except FileNotFoundError:  
        print(f"The file {file_path} does not exist.")  
        return ""  
    except Exception as e:  
        print(f"An error occurred while reading the file: {e}")  
        return ""  
      
    return file_content 

def summarize_file(input_text: str):
    """  
    Summarize the content of a file using OpenAI's GPT.  
  
    :param file_content: The content of the file as a string.  
    :return: A summary of the file content.  
    """  
    # Define the prompt for GPT  
    summary_prompt = f"Summarize the functionality of the following Python file:\n\n{input_text}"

    return query_model(prompt=summary_prompt)

def find_functions_with_no_comments(filename):
    """Parse a Python file to find functions without descriptive comments."""
    with open(filename, "r") as file:
        tree = ast.parse(file.read())

    functions = []
    with open(filename, "r") as file:
        lines = file.readlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_start = node.lineno - 1

            # Check lines above the function definition
            has_comment = False
            for i in range(function_start - 1, -1, -1):
                line = lines[i].strip()

                if line == "":
                    continue
                
                # Check for single-line comment
                if line.startswith("#"):
                    has_comment = True
                # Check for multi-line triple-quote comment
                elif line.startswith('"""') or line.startswith("'''"):
                    # If the triple quote starts and ends on the same line
                    if line.endswith('"""') or line.endswith("'''") and len(line) > 3:
                        has_comment = True
                    else:
                        # Handle multi-line comments that start with triple quotes
                        has_comment = True
                        for j in range(i - 1, -1, -1):
                            if lines[j].strip().endswith('"""') or lines[j].strip().endswith("'''"):
                                break
                    break

                break

            if not has_comment:
                functions.append((node.name, function_start))

    return functions

def read_python_file(file_name):  
    """Read the content of a Python file."""  
    with open(file_name, 'r') as file:  
        return file.read()  
  
def extract_function_content(file_name, fns):  
    with open(file_name, "r") as f:  
        tree = ast.parse(f.read())  
      
    functions = []  
    for node in ast.walk(tree):  
        if isinstance(node, ast.FunctionDef):  
            functions.append(node)  
      
    file_content = read_python_file(file_name)  
    lines = file_content.splitlines()  
      
    extracted_functions = {}  
    for node in functions:  
        function_name = node.name  
        function_start = node.lineno - 1  
        function_end = node.body[-1].lineno  
          
        function_lines = lines[function_start:function_end]  
        extracted_functions[function_name] = "\n".join(function_lines)  

    keys_to_keep = {t[0] for t in fns}  
    filtered_fns = dict((k, v) for k, v in extracted_functions.items() if k in keys_to_keep) 
    return filtered_fns


def generate_comments_for_functions(file_name, functions):
    """Generate comments for functions by querying GPT."""
    new_comments = {}
    extracted_fns = extract_function_content(file_name, functions)
    
    for function_name, function_content in extracted_fns.items():
        # Prepare the prompt with function name and content
        comment_prompt = (
            f"Generate a descriptive but concise single-line comment for the following function. The following function is written in Python, so please precede the comment with the character \"#\". Do not add any other notation, just the comment with the hashtag symbol:\n"
            f"Function Name: {function_name}\n\n"
            f"Function Content:\n{function_content}\n"
        )

        # Query GPT for the comment
        
        comment = query_model(prompt=comment_prompt)
        comment = format_comment(comment)
    
        # Ensure the comment starts with a "#" or triple quotes and properly formatted
        stripped_comment = comment.strip()
        '''
        if not (stripped_comment.startswith("#") or stripped_comment.startswith('"""')):
            comment = "#" + comment
        elif stripped_comment.startswith('"""') and not stripped_comment.endswith('"""'):
            comment = stripped_comment + ' """'
        elif stripped_comment.startswith("#") and not stripped_comment.startswith("# "):
            comment = "# " + stripped_comment[1:]
        '''
        new_comments[function_name] = stripped_comment

    return new_comments
  
def format_comment(comment):  
    """Ensure the comment is properly formatted for Python files."""  
    if comment.startswith('#'):  
        return comment  
    if comment.startswith('"""') or comment.startswith("'''"):  
        return comment  
    # If it's a multi-line comment, use triple quotes  
    if '\n' in comment:  
        return f'"""\n{comment}\n"""'  
    # Otherwise, use a single line comment  
    return f'# {comment}'  
  
import re  
  
def insert_comments_into_file(input_file_path, output_file_path, comments):  
    """Insert comments before the corresponding function definitions and write to a new file."""  
    with open(input_file_path, 'r', encoding='utf-8') as file:  
        lines = file.readlines()  
  
    # Create a pattern to match function definitions  
    function_pattern = re.compile(r"^def\s+(\w+)\s*\(")  
  
    # We will iterate over the lines in reverse to avoid affecting the line numbers of subsequent functions  
    for i in range(len(lines) - 1, -1, -1):  
        match = function_pattern.match(lines[i].strip())  
        if match:  
            function_name = match.group(1)  
            if function_name in comments:  
                comment_lines = comments[function_name].split('\n')  
                lines.insert(i, '\n'.join(comment_lines) + '\n')  
  
    # Write the modified content to the output file  
    with open(output_file_path, 'w', encoding='utf-8') as file:  
        file.writelines(lines)

def comment_file(input_filename, output_filename):
    functions = find_functions_with_no_comments(input_filename)
    new_comments = generate_comments_for_functions(input_filename, functions) 
    insert_comments_into_file(input_filename, output_filename, new_comments) 

if __name__ == "__main__":
    comment_file("src/ex_code.py", "src/ex_code_commented.py")  # Replace with your input Python file

    
