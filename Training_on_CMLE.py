# Distributed training and hyperparameter tuning
# Making the code a Python package
# Using gcloud to submit the training code to Cloud ML Engine

import os

os.environ["BUCKET"] = BUCKET
os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION
os.environ["TFVERSION"] = "1.8"
# %bash
# gcloud config set project $PROJECT
# gcloud config set compute/region $REGION

# %%bash
# if ! gsutil ls | grep -q gs://${BUCKET}/babyweight/preproc; then
#   gsutil mb -l ${REGION} gs://${BUCKET}
#   # copy canonical set of preprocessed files
#   gsutil -m cp -R gs://cloud-training-demos/babyweight gs://${BUCKET}
# fi

# %bash
# gsutil ls gs://${BUCKET}/babyweight/preproc/*-00000*


## Making sure model works in standalone mode (only one training example)
# %bash
# echo "bucket=${BUCKET}"
# rm -rf babyweight_trained
# export PYTHONPATH=${PYTHONPATH}:${PWD}/babyweight
# python -m trainer.task \
#   --bucket=${BUCKET} \
#   --output_dir=babyweight_trained \
#   --job-dir=./tmp \
#   --pattern="00000-of-" --train_examples=1 --eval_steps=1


## Now we can bring it to ML engine
## unique JOBNAME
## before --\ is for ml-engine
## after --\ is for the task.py

# %bash
# OUTDIR=gs://${BUCKET}/babyweight/trained_model
# JOBNAME=babyweight_$(date -u +%y%m%d_%H%M%S)
# echo $OUTDIR $REGION $JOBNAME
# gsutil -m rm -rf $OUTDIR
# gcloud ml-engine jobs submit training $JOBNAME \
#   --region=$REGION \
#   --module-name=trainer.task \
#   --package-path=$(pwd)/babyweight/trainer \
#   --job-dir=$OUTDIR \
#   --staging-bucket=gs://$BUCKET \
#   --scale-tier=STANDARD_1 \
#   --runtime-version=$TFVERSION \
#   -- \
#   --bucket=${BUCKET} \
#   --output_dir=${OUTDIR} \
#   --train_examples=200000

## monitor in ml engine in GC console
## Visualize using tersorboard
from google.datalab.ml import TensorBoard

TensorBoard().start("gs://{}/babyweight/trained_model".format(BUCKET))

for pid in TensorBoard.list()["pid"]:
    TensorBoard().stop(pid)
    print("Stopped TensorBoard with pid {}".format(pid))
