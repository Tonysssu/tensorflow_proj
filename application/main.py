import json
import os

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from google.appengine.api import app_identity

# authenticate
credentials = GoogleCredentials.get_application_default()
api = discovery.build("ml", "v1", credentials=credentials)
project = app_identity.get_application_id()
model_name = os.getenv("MODEL_NAME", "babyweight")
version_name = os.getenv("VERSION_NAME", "ml_on_gcp")


app = Flask(__name__)


def get_prediction(features):
    input_data = {"instances": [features]}
    parent = "projects/%s/models/%s" % (project, model_name)
    prediction = api.projects().predict(body=input_data, name=parent).execute()
    return prediction["predictions"][0]["predictions"][0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/form")
def input_form():
    return render_template("form.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    def gender2str(val):
        genders = {"unknown": "Unknown", "male": "True", "female": "False"}
        return genders[val]

    def plurality2str(val):
        pluralities = {"1": "Single(1)", "2": "Twins(2)", "3": "Triplets(3)"}
        if features["is_male"] == "Unknown" and int(val) > 1:
            return "Multiple(2+)"
        return pluralities[val]

    data = json.loads(request.data.decode())
    mandatory_items = ["baby_gender", "mother_age", "plurality", "gestation_weeks"]
    for item in mandatory_items:
        if item not in data.keys():
            return jsonify({"result": "Set all items."})

    features = {}
    features["key"] = "nokey"
    features["is_male"] = gender2str(data["baby_gender"])
    features["mother_age"] = float(data["mother_age"])
    features["plurality"] = plurality2str(data["plurality"])
    features["gestation_weeks"] = float(data["gestation_weeks"])

    prediction = get_prediction(features)
    return jsonify({"result": "{:.2f} lbs.".format(prediction)})
