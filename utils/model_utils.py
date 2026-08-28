import os
try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("WARNING: TensorFlow is not installed. ML predictions will be disabled.")
import numpy as np
from config import Config

# Global variable to store the loaded model
_model = None

def load_keras_model():
    """
    Load the trained Keras model from disk.
    This should be called once when the app starts.
    """
    global _model
    if _model is None:
        try:
            if tf is None:
                print("Error: TensorFlow is not available. Cannot load the model.")
                return False
                
            if not os.path.exists(Config.MODEL_PATH):
                print(f"Error: Model file not found at {Config.MODEL_PATH}")
                print("Please run 'python training/dummy_model_generator.py' or train the real model.")
                return False
                
            print("Loading model... This might take a few seconds.")
            _model = tf.keras.models.load_model(Config.MODEL_PATH)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    return True

def predict_skin_lesion(preprocessed_image):
    """
    Run inference on a single preprocessed image.
    Returns the predicted class details and probabilities.
    """
    global _model
    if _model is None:
        if not load_keras_model():
            raise Exception("Model is not loaded and could not be loaded.")
            
    # Perform prediction
    predictions = _model.predict(preprocessed_image, verbose=0)
    
    # Extract probabilities for the first (and only) image in batch
    probabilities = predictions[0]
    
    # Get the index of the highest probability
    predicted_class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class_idx])
    
    # Map index to class information
    class_info = Config.CLASS_MAPPING.get(predicted_class_idx, {"code": "unknown", "name": "Unknown"})
    
    # Format all probabilities nicely
    all_probs = []
    for idx, prob in enumerate(probabilities):
        info = Config.CLASS_MAPPING.get(idx, {"code": f"class_{idx}", "name": f"Class {idx}"})
        all_probs.append({
            "code": info["code"],
            "name": info["name"],
            "probability": float(prob),
            "percentage": round(float(prob) * 100, 2)
        })
        
    # Sort probabilities in descending order
    all_probs.sort(key=lambda x: x["probability"], reverse=True)
    
    # Interpret confidence level
    if confidence >= 0.85:
        confidence_level = "High model confidence"
    elif confidence >= 0.50:
        confidence_level = "Moderate model confidence"
    else:
        confidence_level = "Low model confidence"
    
    result = {
        "predicted_class_code": class_info["code"],
        "predicted_class_name": class_info["name"],
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "confidence_level": confidence_level,
        "all_probabilities": all_probs
    }
    
    return result
