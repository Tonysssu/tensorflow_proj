import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION

%%bash
if ! gsutil ls | grep -q gs://${BUCKET}/; then
  gsutil mb -l ${REGION} gs://${BUCKET}
fi

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
    #Using String format to form sql query
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
    """.format(column_name)
    return bq.Query(query).execute().result().to_dataframe()

df2 = get_distinct_values(is_male)
df2.head()
df2.plot(x='is_male', y='num_babies', logy=True, kind='bar');
df2.plot(x='is_male', y='avg_wt', kind='bar');
