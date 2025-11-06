import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# --- 1. DEFINE PARAMETERS AND DATA PATHS ---

# ** CORRECTED PATHS to look inside the 'dataset' folder **
train_dir = 'dataset/train'
validation_dir = 'dataset/test'

# MobileNetV2 works best with larger images than 48x48. 96x96 is a good compromise.
IMG_SIZE = 96
BATCH_SIZE = 64
NUM_CLASSES = 7 # We still have 7 emotions for now

# --- 2. CREATE A POWERFUL DATA AUGMENTATION PIPELINE ---

def cutout(image):
    """Applies the Cutout augmentation technique."""
    h, w, _ = image.shape
    scale = 0.25
    mask_h = int(h * scale)
    mask_w = int(w * scale)
    x = np.random.randint(0, w - mask_w)
    y = np.random.randint(0, h - mask_h)
    image[y:y+mask_h, x:x+mask_w, :] = 128
    return image

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    preprocessing_function=cutout, 
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical'
)

# --- 3. BUILD THE MODEL USING TRANSFER LEARNING ---

base_model = MobileNetV2(
    include_top=False, 
    weights='imagenet', 
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("--- Model Architecture ---")
model.summary()

# --- 4. COMPILE AND TRAIN THE MODEL ---

model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("\n--- Starting Training ---")
epochs = 25

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# --- 5. SAVE THE FINAL, FINE-TUNED MODEL ---

model.save('emotion_finetuned_mobilenetv2.h5')
print("\n--- Training complete! Model saved as 'emotion_finetuned_mobilenetv2.h5' ---")