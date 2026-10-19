from PIL import Image
import os

def preprocess_image(input_path, output_path, size=(150,150)):
    img = Image.open(input_path)
    img = img.resize(size)
    img.save(output_path)
