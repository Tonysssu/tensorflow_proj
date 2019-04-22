import os

os.environ["BUCKET"] = BUCKET
os.environ["PROJECT"] = PROJECT
os.environ["REGION"] = REGION

# %%bash
# if ! gsutil ls | grep -q gs://${BUCKET}/; then
#   gsutil mb -l ${REGION} gs://${BUCKET}
# fi

# Create SQL query using natality data after the year 2000
query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  ABS(FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING)))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""

# Call BigQuery and examine in dataframe
import google.datalab.bigquery as bq

df = bq.Query(query + " LIMIT 100").execute().result().to_dataframe()
df.head()


def get_distinct_values(column_name):
    # Using String format to form sql query
    query = """
    SELECT
        {0},
        COUNT(1) as num_babies,
        AVG(weight_pounds) as avg_wt
    FROM
      publicdata.samples.natality
    WHERE year > 2000
    GROUP BY
    {0}
    """.format(
        column_name
    )
    return bq.Query(query).execute().result().to_dataframe()


df2 = get_distinct_values(is_male)
df2.head()
df2.plot(x="is_male", y="num_babies", logy=True, kind="bar")
df2.plot(x="is_male", y="avg_wt", kind="bar")


##Creating sampled data
### Create ML dataset by sampling using bigquery
# Create SQL query using natality data after the year 2000
import google.datalab.bigquery as bq

query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  ABS(FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING)))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""

# Call BigQuery but GROUP BY the hashmonth and see number of records for each group to
# get the correct train and evaluation percentages
df = (
    bq.Query(
        "SELECT hashmonth, COUNT(weight_pounds) AS num_babies FROM ("
        + query
        + ") GROUP BY hashmonth"
    )
    .execute()
    .result()
    .to_dataframe()
)
print("There are {} unique hashmonths.".format(len(df)))
df.head()

# Added the RAND() so that we can now subsample from each of the hashmonths to get approximately the record counts we want
trainQuery = (
    "SELECT * FROM (" + query + ") WHERE MOD(hashmonth, 4) < 3 AND RAND() < 0.0005"
)
evalQuery = (
    "SELECT * FROM (" + query + ") WHERE MOD(hashmonth, 4) = 3 AND RAND() < 0.0005"
)
traindf = bq.Query(trainQuery).execute().result().to_dataframe()
evaldf = bq.Query(evalQuery).execute().result().to_dataframe()
print(
    "There are {} examples in the train dataset and {} in the eval dataset".format(
        len(traindf), len(evaldf)
    )
)
traindf.head()
traindf.describe()

# Preprocessing data
import pandas as pd


def preprocess(df):
    # clean up data we don't want to train on
    # in other words, users will have to tell us the mother's age
    # otherwise, our ML service won't work.
    # these were chosen because they are such good predictors
    # and because these are easy enough to collect
    df = df[df.weight_pounds > 0]
    df = df[df.mother_age > 0]
    df = df[df.gestation_weeks > 0]
    df = df[df.plurality > 0]

    # modify plurality field to be a string
    twins_etc = dict(
        zip(
            [1, 2, 3, 4, 5],
            [
                "Single(1)",
                "Twins(2)",
                "Triplets(3)",
                "Quadruplets(4)",
                "Quintuplets(5)",
            ],
        )
    )
    df["plurality"].replace(twins_etc, inplace=True)

    # now create extra rows to simulate lack of ultrasound
    nous = df.copy(deep=True)
    nous.loc[nous["plurality"] != "Single(1)", "plurality"] = "Multiple(2+)"
    nous["is_male"] = "Unknown"

    return pd.concat([df, nous])


traindf.to_csv("train.csv", index=False, header=False)
evaldf.to_csv("eval.csv", index=False, header=False)

# %bash
# wc -l *.csv
# head *.csv
# tail *.csv
