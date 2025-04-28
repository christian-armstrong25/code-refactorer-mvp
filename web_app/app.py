# app.py
import os
import shutil
import zipfile
from flask import Flask, render_template, request, send_file, redirect, Response, stream_with_context
from pathlib import Path

from run_pipeline import run_full_pipeline_generator  # Import the generator function

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ZIP_FOLDER = 'zipped'
ZIP_FILE = 'files.zip'

# Ensure folders exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, ZIP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def clear_folders():
    """Clear content in specified folders."""
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, ZIP_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

def generate_tree(path, prefix=""):
    output = []
    path = Path(path)

    if path.is_dir() and path.name == "processed":
        for item in sorted(path.iterdir()):
            output.extend(generate_tree(item, prefix))
    elif path.is_file():
        output.append(f"{prefix}|-- {path.name}")
    elif path.is_dir():
        output.append(f"{prefix}|-- {path.name}/")
        prefix += "|   "
        for item in sorted(path.iterdir()):
            output.extend(generate_tree(item, prefix))
    
    return output

def zip_directory(source_dir, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, source_dir))

@app.route('/', methods=['GET'])
def index():
    # Clear folders at the start when the user opens the site
    clear_folders()
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    def generate():
        # Handle uploaded files and directories
        if request.files:
            for key in request.files:
                file = request.files[key]
                
                # Use the full relative path provided by the file key
                relative_path = key.replace("\\", "/")
                file_path = os.path.join(UPLOAD_FOLDER, relative_path)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save the file
                file.save(file_path)
        
        # Determine the input_directory to pass to run_full_pipeline_generator
        entries = os.listdir(UPLOAD_FOLDER)
        if len(entries) == 1 and os.path.isdir(os.path.join(UPLOAD_FOLDER, entries[0])):
            input_directory = os.path.join(UPLOAD_FOLDER, entries[0])
        else:
            input_directory = UPLOAD_FOLDER

        # Run the pipeline and yield status updates
        for status in run_full_pipeline_generator(input_directory, PROCESSED_FOLDER):
            yield status + '\n'
        
        # Generate tree after processing
        tree = "\n".join(generate_tree(PROCESSED_FOLDER))
        
        # Zip the processed files
        zip_output_path = os.path.join(ZIP_FOLDER, ZIP_FILE)
        zip_directory(PROCESSED_FOLDER, zip_output_path)
        
        # Clean up uploaded files
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)

        yield 'PROCESSING_COMPLETE\n' + tree

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/download', methods=['GET'])
def download():
    zip_output_path = os.path.join(ZIP_FOLDER, ZIP_FILE)
    if os.path.exists(zip_output_path):
        return send_file(zip_output_path, as_attachment=True)
    else:
        return "No file available for download", 404

@app.route('/reset', methods=['GET'])
def reset():
    # Clean up uploaded and processed files
    clear_folders()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=False)
