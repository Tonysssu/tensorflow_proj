import shutil
import numpy as np
import tensorflow as tf

# Determine CSV, label, and key column
CSV_COLUMNS = "weight_pounds,is_male,mother_age,plurality,gestation_weeks,key".split(
    ","
)
LABEL_COLUMN = "weight_pounds"
KEY_COLUMN = "key"

# Set default values for each CSV column
DEFAULTS = [[0.0], ["null"], [0.0], ["null"], [0.0], ["nokey"]]
TRAIN_STEPS = 1000

##TASK 2
# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dataset(filename_pattern, mode, batch_size=512):
    def _input_fn():
        def decode_csv(line_of_text):
            # 1: Use tf.decode_csv to parse the provided line
            columns = tf.decode_csv(line_of_text, record_defaults=DEFAULTS)

            # 2: Make a Python dict.  The keys are the column names, the values are from the parsed data
            features = dict(zip(CSV_COLUMNS, columns))

            # 3: Return a tuple of features, label where features is a Python dict and label a float
            label = features.pop(LABEL_COLUMN)

            return features, label

        # 4: Use tf.gfile.Glob to create list of files that match pattern
        file_list = tf.gfile.Glob(filename_pattern)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(
            decode_csv
        )  # Read text file # Transform each elem by applying decode_csv fn

        # 5: In training mode, shuffle the dataset and repeat indefinitely
        #                (Look at the API for tf.data.dataset shuffle)
        #          The mode input variable will be tf.estimator.ModeKeys.TRAIN if in training mode
        #          Tell the dataset to provide data in batches of batch_size
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        # This will now return batches of features, label
        return dataset

    return _input_fn


##TASK 3
# Define feature columns
is_male_cols = tf.feature_column.categorical_column_with_vocabulary_list(
    "is_male", ["True", "False", "null"]
)
plurality_cols = tf.feature_column.categorical_column_with_vocabulary_list(
    "plurality", ["1", "2", "3", "4", "5"]
)

wide_cols = [is_male_cols, plurality_cols]

mother_age_cols = tf.feature_column.numeric_column("mother_age")
gestation_weeks_cols = tf.feature_column.numeric_column("gestation_weeks")

deep_cols = [mother_age_cols, gestation_weeks_cols]

##TASK 4
# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    feature_placeholders = {
        "is_male": tf.placeholder(tf.string, [None]),
        "mother_age": tf.placeholder(tf.float32, [None]),
        "plurality": tf.placeholder(tf.string, [None]),
        "gestation_weeks": tf.placeholder(tf.float32, [None]),
    }
    features = {
        key: tf.expand_dims(tensor, -1) for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


## TASK 5
# Create estimator to train and evaluate
def train_and_evaluate(output_dir):
    EVAL_INTERVAL = 300
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=EVAL_INTERVAL, keep_checkpoint_max=3
    )
    # 1: Create estimator
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir=output_dir,
        linear_feature_columns=wide_cols,
        dnn_feature_columns=deep_cols,
        dnn_hidden_units=[100, 50],
    )

    train_spec = tf.estimator.TrainSpec(
        # 2: Call read_dataset passing in the training CSV file and the appropriate mode
        input_fn=read_dataset("train.csv", mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=TRAIN_STEPS,
    )

    exporter = tf.estimator.LatestExporter("exporter", serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        # 3: Call read_dataset passing in the evaluation CSV file and the appropriate mode
        input_fn=read_dataset("eval.csv", mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        start_delay_secs=60,  # start evaluating after N seconds
        throttle_secs=EVAL_INTERVAL,  # evaluate every N seconds
        exporters=exporter,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


## TASK 6
# Run the model
shutil.rmtree("babyweight_trained", ignore_errors=True)  # start fresh each time
tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file
train_and_evaluate("babyweight_trained")

## TASK 7
from google.datalab.ml import TensorBoard

TensorBoard().start("./babyweight_trained")
for pid in TensorBoard.list()["pid"]:
    TensorBoard().stop(pid)
    print("Stopped TensorBoard with pid {}".format(pid))
