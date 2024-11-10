from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from typing import Dict, Any
import logging
from main import AllInOne
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK data
try:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("stopwords")
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
OUTPUT_FOLDER = 'outputs'

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

# @app.route('/upload', methods=['POST'])
# def upload_file() -> Dict[str, str]:
#     """Upload a CSV file"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    

@app.route('/process', methods=['POST'])
def process_data() -> Dict[str, Any]:
    """
    Process CSV data and return results
    
    Expected form data:
    - file: CSV file
    - column: Column name containing company names
    - question: Question to be answered
    - max_workers (optional): Maximum number of parallel workers
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
            
        column = request.form.get('column')
        if not column:
            return jsonify({'error': 'Column name not provided'}), 400
            
        question = request.form.get('question')
        if not question:
            return jsonify({'error': 'Question not provided'}), 400
            
        max_workers = int(request.form.get('max_workers', 3))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logging.info(f"Processing file: {filename}")
        
        # Process the data
        processor = AllInOne(
            csv_path=filepath,
            column=column,
            question=question,
            max_workers=max_workers
        )
        
        # Execute processing
        processor()
        
        # Save results
        output_filename = f"processed_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        processor.save_results(output_path)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'message': 'Processing completed',
            'output_file': output_filename
        })
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        # Clean up uploaded file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_result(filename: str) -> Any:
    """Download processed results"""
    try:
        if not os.path.exists(os.path.join(OUTPUT_FOLDER, filename)):
            return jsonify({'error': 'File not found'}), 404
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5001)
