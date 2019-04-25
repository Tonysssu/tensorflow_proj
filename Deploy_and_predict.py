import os

os.environ["BUCKET"] = BUCKET
os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION
os.environ["TFVERSION"] = "1.9"

# %%bash
# if ! gsutil ls | grep -q gs://${BUCKET}/babyweight/trained_model; then
#   gsutil mb -l ${REGION} gs://${BUCKET}
#   # copy canonical model
#   gsutil -m cp -R gs://cloud-training-demos/babyweight/trained_model gs://${BUCKET}/babyweight/trained_model
# fi

# MODEL_NAME="babyweight"
# MODEL_VERSION="ml_on_gcp"
# MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/babyweight/trained_model/export/exporter/ | tail -1)
# echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
# #gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
# #gcloud ml-engine models delete ${MODEL_NAME}
# gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
# gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION

# online prediction
from oauth2client.client import GoogleCredentials
import requests
import json

MODEL_NAME = "babyweight"
MODEL_VERSION = "ml_on_gcp"

token = GoogleCredentials.get_application_default().get_access_token().access_token
api = "https://ml.googleapis.com/v1/projects/{}/models/{}/versions/{}:predict".format(
    PROJECT, MODEL_NAME, MODEL_VERSION
)
headers = {"Authorization": "Bearer " + token}
data = {
    "instances": [
        {
            "key": "b1",
            "is_male": "True",
            "mother_age": 26.0,
            "plurality": "Single(1)",
            "gestation_weeks": 39,
        },
        {
            "key": "g1",
            "is_male": "False",
            "mother_age": 29.0,
            "plurality": "Single(1)",
            "gestation_weeks": 38,
        },
        {
            "key": "b2",
            "is_male": "True",
            "mother_age": 26.0,
            "plurality": "Triplets(3)",
            "gestation_weeks": 39,
        },
        {
            "key": "u1",
            "is_male": "Unknown",
            "mother_age": 29.0,
            "plurality": "Multiple(2+)",
            "gestation_weeks": 38,
        },
    ]
}
response = requests.post(api, json=data, headers=headers)
print(response.content)

# batch prediction
# %writefile inputs.json
# {"key": "b1", "is_male": "True", "mother_age": 26.0, "plurality": "Single(1)", "gestation_weeks": 39}
# {"key": "g1", "is_male": "False", "mother_age": 26.0, "plurality": "Single(1)", "gestation_weeks": 39}

# %bash
# INPUT=gs://${BUCKET}/babyweight/batchpred/inputs.json
# OUTPUT=gs://${BUCKET}/babyweight/batchpred/outputs
# gsutil cp inputs.json $INPUT
# gsutil -m rm -rf $OUTPUT
# gcloud ml-engine jobs submit prediction babypred_$(date -u +%y%m%d_%H%M%S) \
#   --data-format=TEXT --region ${REGION} \
#   --input-paths=$INPUT \
#   --output-path=$OUTPUT \
#   --model=babyweight --version=ml_on_gcp
