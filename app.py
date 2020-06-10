import json
import os
from flask import Flask, render_template, request, redirect, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
UPLOAD_FOLDER = "static/logfiles/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(["json"])


# @app.route("/process", methods=["POST"])
# def process_sample():
#     if request.method == "POST":
#         filename = request.form["sample_name"]
#         inference_engine = ImagePipeline()
#     return jsonify({"image_name": filename, "table_name": "tablenamehere"})


@app.route("/")
def index():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8029")
