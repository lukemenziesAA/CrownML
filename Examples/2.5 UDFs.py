# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Pandas UDF for Training and Scoring*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md 
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * Use apply in Pandas for Training
# MAGIC * Use apply in Pandas for Scoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installing Libraries
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import max, min, year, lit, month, col, struct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from xgboost import XGBRegressor
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a parquet file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

data = (
    spark.read.format("parquet")
    .load("dbfs:/FileStore/tables/household_power_consumption.parquet")
    .withColumn("Year", year("Datetime"))
)

# COMMAND ----------

data.display()

# COMMAND ----------

label = "Voltage"
date_col = "datetime"

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ###Set Experiment
# MAGIC Create and experiment to centralise the runs. If the experiment already exists, set the extisting one. 

# COMMAND ----------

try:
    experiment_id = mlflow.create_experiment('/Workspace/Shared/udf_exp')
except:
    experiment_id = mlflow.set_experiment('/Shared/udf_exp').experiment_id

# COMMAND ----------

model = XGBRegressor()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Function
# MAGIC Create a training routine that trains a model and returns the run_id, along with the partition key. This partitiion key is what the dataset is grouped on, such that there's a model for each partition. Ensuring that a Pandas DataFrame is returned

# COMMAND ----------

def training_routine(data):
    mlflow.xgboost.autolog()
    yearv = data.Year[0]
    X = data.drop([label, date_col], axis=1)
    y = data[[label]].fillna(0)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{yearv}") as run:
        run_id = run.info.run_id
        model.fit(X, y.values.flatten())
    return pd.DataFrame([[run_id, int(yearv)]], columns=["run_id", "Year"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create output schema
# MAGIC Create the output schema that returns the run_id and the partition key

# COMMAND ----------

schema = StructType(
    [
        StructField("run_id", StringType(), True),
        StructField("Year", IntegerType(), True),
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform groupby
# MAGIC Perform groupby on the partition key to train each partition

# COMMAND ----------

out_df = data.groupBy('Year').applyInPandas(training_routine, schema).cache()

# COMMAND ----------

out_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scoring with UDFs
# MAGIC You can also score with UDFs directly using MLflow thorugh the spark_udf functionality. This is only available with PyFunc models. 

# COMMAND ----------


model_path = f"runs:/{out_df.limit(1).collect()[0].run_id}/model"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_path)
data.withColumn('predictions', loaded_model(struct(*map(col, data.columns))))

# COMMAND ----------

data_with_mod = data.join(out_df, on=['Year']).withColumn("Prediction", lit(0.00))

# COMMAND ----------

display(data_with_mod)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scoring using Pandas UDFs
# MAGIC Once all the models have been trained, the following routine below can be used to score on another dataset. Passing the run id to the appropriate partition allows the use of MLflow to load the model in and score on the data. The data is then returned as a Pandas DataFrame. 

# COMMAND ----------

columns = ['Global_active_power',
 'Global_reactive_power',
 'Global_intensity',
 'Sub_metering_1',
 'Sub_metering_2',
 'Sub_metering_3',
 'Year'
 ]

def inference(data):
    run_id = data.run_id[0]
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)
    pred = model.predict(data[columns])
    data['Prediction'] = pred
    return data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scoring using groupby
# MAGIC Use the groupby and applyInPandas, passing the function through to score the data. 

# COMMAND ----------

final_out = data_with_mod.groupBy("Year").applyInPandas(inference, data_with_mod.schema).cache()

# COMMAND ----------

final_out.display()

# COMMAND ----------

