import os
import cv2
import uuid
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

def generate_gradcam(img_array, model, original_img_path, save_dir, filename_prefix):
    """Generate Grad-CAM heatmap and overlay it on the original image."""
    try:
        # Find the base model if nested (like in transfer learning)
        target_model = model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                target_model = layer
                break
                
        # Find the last convolutional layer
        last_conv_layer_name = None
        for layer in reversed(target_model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_layer_name = layer.name
                    break
            except Exception:
                pass

        if not last_conv_layer_name:
            print("Could not find a valid Conv2D layer for Grad-CAM.")
            return None

        # Create a model that outputs the conv layer and the target model output
        grad_model = tf.keras.models.Model(
            [target_model.inputs], [target_model.get_layer(last_conv_layer_name).output, target_model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # If the model is nested, pass the predictions through the rest of the layers
            if target_model != model:
                x = predictions
                started = False
                for layer in model.layers:
                    if started:
                        x = layer(x)
                    elif layer == target_model:
                        started = True
                final_predictions = x
            else:
                final_predictions = predictions

            # Target the class with the highest probability
            class_idx = tf.argmax(final_predictions[0])
            loss = final_predictions[:, class_idx]

        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel by its gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ tf.expand_dims(pooled_grads, axis=-1)
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize heatmap
        max_val = tf.math.reduce_max(heatmap)
        if max_val != 0:
            heatmap = heatmap / max_val
        heatmap = heatmap.numpy()

        # Load original image and overlay heatmap
        img = cv2.imread(original_img_path)
        if img is None: return None
        
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed_img = cv2.addWeighted(heatmap_colored, 0.4, img, 0.6, 0)
        
        gradcam_filename = f"gradcam_{filename_prefix}"
        save_path = os.path.join(save_dir, gradcam_filename)
        cv2.imwrite(save_path, superimposed_img)
        
        return gradcam_filename
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

def predict_skin_lesion(preprocessed_image, filepath=None):
    """
    Run inference on a single preprocessed image.
    Returns the predicted class details, probabilities, and optionally Grad-CAM.
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
    
    # Generate Grad-CAM if filepath is provided
    if filepath and os.path.exists(filepath):
        filename_prefix = os.path.basename(filepath)
        save_dir = os.path.dirname(filepath)
        gradcam_filename = generate_gradcam(preprocessed_image, _model, filepath, save_dir, filename_prefix)
        if gradcam_filename:
            result['gradcam_filename'] = gradcam_filename
            
    return result
