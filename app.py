from flask import Flask, render_template, request, send_file
from colorize import colorize_image
from super_resolution import super_resolve
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'  # Folder for uploading images
RESULT_FOLDER = 'results/'  # Folder for saving results

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image colorization
@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Colorize the image
        output_path = os.path.join(RESULT_FOLDER, 'colorized_' + file.filename)
        colorized_image_path = colorize_image(file_path, output_path)
        
        return send_file(colorized_image_path, mimetype='image/jpeg')

from PIL import Image

@app.route('/super_resolve', methods=['POST'])
def super_resolve_image():
    if 'image' not in request.files:
        return "No file part", 400  # Return a bad request error if 'image' is missing
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400  # Return error if no file is selected
    
    # Proceed with processing the image
    image = Image.open(file).convert("RGB")
    super_resolved_image = super_resolve(image)
    
    output_path = os.path.join("uploads", "super_resolved_image.png")
    super_resolved_image.save(output_path)
    
    return send_file(output_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
