import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret_key_12345')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model settings
    MODEL_PATH = os.path.join('model', 'trained_model.keras')
    IMAGE_SIZE = (224, 224)
    
    # Class mapping for HAM10000 dataset
    # Make sure this matches the order of the model's output neurons
    CLASS_MAPPING = {
        0: {"code": "akiec", "name": "Actinic Keratoses / Intraepithelial Carcinoma"},
        1: {"code": "bcc", "name": "Basal Cell Carcinoma"},
        2: {"code": "bkl", "name": "Benign Keratosis-like Lesions"},
        3: {"code": "df", "name": "Dermatofibroma"},
        4: {"code": "mel", "name": "Melanoma"},
        5: {"code": "nv", "name": "Melanocytic Nevi"},
        6: {"code": "vasc", "name": "Vascular Lesions"}
    }
