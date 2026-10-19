
from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from predict import predict_image_class

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template(
            'index.html', error="Please select an image."
        )

    file = request.files['file']

    if file.filename == '':
        return render_template(
            'index.html', error="Please select an image."
        )

    if not allowed_file(file.filename):
        return render_template(
            'index.html', error="Upload a JPG, JPEG, or PNG image."
        )

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # Predict the actual image class using your trained model
        prediction = predict_image_class(filepath)

        return render_template(
            'index.html',
            image_url=url_for('static', filename='uploads/' + filename),
            prediction=prediction
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Prediction failed: {e}"
        )


if __name__ == '__main__':
    app.run(debug=True)
