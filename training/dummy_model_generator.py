import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

def generate_dummy_model():
    """
    Generates a dummy untrained Keras model with the same architecture 
    as the expected final model (MobileNetV2 base + 7-class head).
    This allows testing the Flask application UI before the real model is trained.
    """
    print("Generating dummy model for UI testing...")
    
    # Create the model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Input tensor (224, 224, 3)
    inputs = Input(shape=(224, 224, 3))
    
    # MobileNetV2 base (using randomly initialized weights for the dummy model)
    # Note: We use weights=None so it doesn't try to download weights on restrictive networks,
    # it's just a dummy model anyway.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None 
    )
    
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    
    # Output layer for 7 classes with softmax activation
    outputs = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model (required before saving in some TF versions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_path = os.path.join('model', 'trained_model.keras')
    model.save(model_path)
    
    print(f"Dummy model successfully saved to {model_path}")
    print("You can now run 'python app.py' to test the Flask application.")
    print("WARNING: This model produces random predictions. Replace it with the real model trained on Colab.")

if __name__ == "__main__":
    generate_dummy_model()
