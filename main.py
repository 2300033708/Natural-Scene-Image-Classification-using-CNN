
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_image_class

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if file is None or file.filename == "":
            error = "Please select an image."

        elif not allowed_file(file.filename):
            error = "Upload a JPG, JPEG, or PNG image."

        else:
            filename = secure_filename(file.filename)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"], filename
            )

            try:
                file.save(filepath)
                prediction = predict_image_class(filepath)
                image_path = "/" + filepath.replace("\\", "/")

            except Exception as e:
                error = f"Prediction failed: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        error=error
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Please open the home page at / to upload an image."

    return index()


if __name__ == "__main__":
    app.run(debug=True)
