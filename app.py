import os
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from config import Config
from utils.image_processing import allowed_file, preprocess_image
from utils.model_utils import load_keras_model, predict_skin_lesion

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Try to load the model on startup
print("Initializing Skin Disease Classification Web App...")
model_loaded = load_keras_model()
if not model_loaded:
    print("WARNING: Model could not be loaded. Predictions will fail.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Check if model is loaded
    if not load_keras_model():
        return jsonify({"error": "Model is not loaded on the server. Please contact administrator."}), 500

    # 2. Check if file is present in request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    # 3. Check if file is empty
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
        
    # 4. Validate and process the file
    if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
        try:
            # Generate a secure, unique filename to prevent collisions and directory traversal attacks
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(filepath)
            
            # 5. Preprocess the image
            preprocessed_img = preprocess_image(filepath, target_size=app.config['IMAGE_SIZE'])
            if preprocessed_img is None:
                return jsonify({"error": "Failed to process the image. The file might be corrupted."}), 500
                
            # 6. Run Prediction
            result = predict_skin_lesion(preprocessed_img)
            
            # 7. Add image path to result for preview
            result['image_url'] = url_for('static', filename=f'uploads/{unique_filename}')
            
            # Return JSON result for AJAX handling
            return jsonify(result)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500
    else:
        return jsonify({"error": f"Allowed file types are {', '.join(app.config['ALLOWED_EXTENSIONS'])}"}), 400

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File is too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def page_not_found(error):
    return render_template('index.html'), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
