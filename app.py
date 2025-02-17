import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from transformers import pipeline
from PyPDF2 import PdfReader  # or use any other library to extract text from PDF

app = Flask(__name__)

# Set the upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Load the zero-shot classification pipeline
resume_classifier = pipeline("zero-shot-classification")

def allowed_file(filename):
    """Check if the uploaded file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    """Extract text from the uploaded PDF file."""
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')  # Display the resume upload form

@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume uploads and process them using AI."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_path)

        # Define categories for the classification
        categories = ["Software Engineer", "Data Scientist", "DevOps Engineer", "Project Manager"]

        # Use the zero-shot classification pipeline with the extracted text and categories
        result = resume_classifier(extracted_text, candidate_labels=categories)

        # Pass the result to the HTML template for display
        return render_template('result.html', filename=filename, 
                               predicted_role=result['labels'][0], 
                               confidence=result['scores'][0])

    return jsonify({"error": "Invalid file format. Only PDFs are allowed."}), 400

if __name__ == '__main__':
    app.run(debug=True)
