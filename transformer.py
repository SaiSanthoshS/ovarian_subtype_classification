import os
import random
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import layers
from vit_keras import vit

# Set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Define paths to your data
data_dir = 'path/to/your/dataset'

# Data preprocessing
batch_size = 32
img_size = (224, 224)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=seed
)

# Load pre-trained Vision Transformer (ViT) model
base_model = vit.vit_b16(
    image_size=img_size + (3,),
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# Build a custom model on top of ViT
model = tf.keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # Assuming 5 subtypes
])

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Evaluate the model on test set if available
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     'path/to/test/dataset',
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical'
# )
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test Accuracy: {test_acc}')

# Save the model
model.save('vit_ovarian_subtype_model.h5')
