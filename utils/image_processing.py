import os
from PIL import Image
import numpy as np

def allowed_file(filename, allowed_extensions):
    """Check if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image to match the requirements of the chosen model.
    For MobileNetV2, we resize to 224x224 and scale pixels to [-1, 1].
    """
    try:
        # Open image using Pillow
        img = Image.open(image_path)
        
        # Convert to RGB in case of PNG with alpha channel or Grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Expand dimensions to create a batch of 1: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # MobileNetV2 preprocessing: scale pixels between -1 and 1
        # img_batch = (img_batch / 127.5) - 1.0
        # (Alternatively, we can use tf.keras.applications.mobilenet_v2.preprocess_input, 
        # but doing it manually avoids heavy dependency inside the preprocessing function if desired).
        # We'll use manual scaling here to match standard MobileNetV2 preprocess
        img_batch = (img_batch / 127.5) - 1.0
        
        return img_batch
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None
