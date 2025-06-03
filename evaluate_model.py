import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = 'resnet_model.h5'
DATA_PATH = 'processed_data/test'
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

print("✅ Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\n✅ Classification Report")
print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
