
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join("saved_models", "intel_model.h5")

# Keep these labels in the same order as your training classes.
CLASS_NAMES = [
    "building",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

model = load_model(MODEL_PATH)


def predict_image_class(img_path):
    # Use the same image size as your model's training input.
    img = image.load_img(img_path, target_size=(150, 150))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))

    if predicted_index >= len(CLASS_NAMES):
        raise ValueError("Model output does not match the class names.")

    confidence = float(predictions[0][predicted_index]) * 100

    return (
        f"{CLASS_NAMES[predicted_index]} "
        f"({confidence:.2f}% confidence)"
    )
