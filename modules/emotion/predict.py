import os
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "surprise",
    "fear",
    "disgust",
    "happy",
    "sad",
    "angry",
    "neutral",
]

def load_model(model_path):
    """Loads the TFLite model from the given path."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # Attach dummy input_shape so page.py technical details don't break
        interpreter.input_shape = interpreter.get_input_details()[0]['shape']
        return interpreter
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def preprocess_image(image: Image.Image, model):
    """
    Automatically resizes the image to whatever size
    the loaded model expects.
    """
    _, height, width, channels = model.input_shape

    image = image.convert("RGB")
    image = image.resize((width, height))

    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

def predict_emotion(model, image):
    """
    Predicts the emotion of a preprocessed image using TFLite.
    Returns (emotion_label, confidence_percentage, full_prediction_array).
    """
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])[0]
    
    idx = np.argmax(prediction)
    emotion = CLASS_NAMES[idx]
    confidence = float(prediction[idx]) * 100

    return emotion, confidence, prediction
