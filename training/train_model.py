import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration
DATASET_DIR = '../dataset'
CSV_PATH = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
MODEL_SAVE_PATH = '../model/trained_model.keras'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 1. Load Metadata
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Metadata CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Loaded metadata with {len(df)} records.")

# 2. Find Images
print("Scanning for image files in the dataset directory...")
image_paths = glob.glob(os.path.join(DATASET_DIR, '**', '*.jpg'), recursive=True)
image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}

if len(image_dict) == 0:
    raise FileNotFoundError("No .jpg images found in the dataset directory. Please make sure to extract the images!")

# Map image_id to actual file path
df['image_path'] = df['image_id'].map(image_dict)
df = df.dropna(subset=['image_path'])  # Remove rows where image was not found
print(f"Found {len(df)} matching images for the dataset.")

# 3. Prepare Labels
# Classes based on your config.py: 
# 0: akiec, 1: bcc, 2: bkl, 3: df, 4: mel, 5: nv, 6: vasc
class_mapping = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}
df['label'] = df['dx'].map(class_mapping)

# 4. Train/Val Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 5. Data Generator
def create_dataset(dataframe):
    paths = dataframe['image_path'].values
    labels = dataframe['label'].values
    
    def generator():
        for path, label in zip(paths, labels):
            img = Image.open(path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)
            # MobileNetV2 preprocessing: scale between -1 and 1
            img_array = (img_array / 127.5) - 1.0
            yield img_array, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)

# 6. Build Model
print("Building MobileNetV2 model...")
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(7, activation='softmax')(x)  # 7 classes

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7. Train Model
print("Starting training...")
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=3, restore_best_weights=True)
]

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"Training complete! Model saved to {MODEL_SAVE_PATH}")
