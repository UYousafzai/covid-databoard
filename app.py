import json
import os
from flask import Flask, render_template, request, redirect, send_file, jsonify
from werkzeug.utils import secure_filename
from covidmodel import get_optimized_params, simulate

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

default_params = {}

@app.route("/")
def index():
    return render_template("dashboard.html", data=retrieve_data("germany", default_params["germany"]))


def retrieve_data(country, params):
    keys = ['dates', 'S', 'E', 'H', 'I', 'C', 'R', 'D', 'Total Cases', 'Total CFR', 'Daily CFR', 'Beds', 'Actual Cases', 'Actual Deaths']

    data_list = simulate(country, params)

    data = {}
    for key, value in zip(keys, data_list):
        data[key] = value
    data['params'] = list(map(lambda x: round(x,4), params))
    data['country'] = country
    return data

@app.route("/get_data", methods=['POST'])
def get_data():
    print(request.form)
    beds = int(request.form['Beds'])
    R0_st = float(request.form['R0-Start'])
    k = float(request.form['k'])
    x0 = float(request.form['x0'])
    R0_end = float(request.form['R0-End'])

    country = request.form['countries']

    if R0_st == 0:
        return render_template("dashboard.html", data=retrieve_data(country, default_params[country]))
    else:
        return render_template("dashboard.html", data=retrieve_data(country, (beds, R0_st, k, x0, R0_end)))


if __name__ == "__main__":   
    countries = ['germany', 'italy', 'ireland']
    for country in countries:
        default_params[country] = get_optimized_params(country)
        
    app.run(host="0.0.0.0", port="8029")
