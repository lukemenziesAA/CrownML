# Databricks notebook source
# MAGIC %md
# MAGIC # Reading and Writing Data in Databricks
# MAGIC
# MAGIC This notebook provides examples of how to read from and write to different data formats in Databricks using Spark. Each section will demonstrate a different format and related options.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading Data
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading CSV Files
# MAGIC
# MAGIC Load a CSV file into a DataFrame with headers and schema inference enabled.
# MAGIC
# MAGIC

# COMMAND ----------

## TASK ONE

asa_dataset_name = "dbfs:/databricks-datasets/asa/small/small.csv"
## Read in the CSV iris_dataset_name with the following settings:
##  - with headers
##  - with schema

df_csv = "ENTER CODE HERE"

df_csv.show(5)


# COMMAND ----------

# DBTITLE 1,TASK ONE ANSWER. Reveal when ready.
asa_dataset_name = "dbfs:/databricks-datasets/asa/small/small.csv"

df_csv = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(asa_dataset_name)

df_csv.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading JSON Files
# MAGIC
# MAGIC Load a JSON file into a DataFrame with multiline records.
# MAGIC

# COMMAND ----------

## TASK TWO
people_dataset_name = "/databricks-datasets/samples/people"
## Read in the JSON people_dataset_name

df_json = "ENTER CODE HERE"

df_json.show(5)

# COMMAND ----------

# DBTITLE 1,TASK TWO ANSWER. Reveal when ready.
people_dataset_name = "/databricks-datasets/samples/people"

df_json = spark.read.format("json") \
    .load(people_dataset_name)

df_json.show(5)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### Reading Parquet Files
# MAGIC
# MAGIC Load a Parquet file into a DataFrame.
# MAGIC

# COMMAND ----------


## TASK THREE

parquet_dataset_name = "/databricks-datasets/learning-spark-v2/loans/loan-risks.snappy.parquet"

## Read in the PARQUET parquet_dataset_name with the following settings:
##  - merge schema (use documentation)

df_parquet = "ENTER CODE HERE"

df_parquet.show(5)



# COMMAND ----------

# DBTITLE 1,TASK THREE ANSWER. Reveal when ready.
parquet_dataset_name = "/databricks-datasets/learning-spark-v2/loans/loan-risks.snappy.parquet"

df_parquet = spark.read.format("parquet") \
    .option("mergeSchema", "true") \
    .load(parquet_dataset_name)

df_parquet.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing Data

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### Writing CSV Files
# MAGIC
# MAGIC Write a DataFrame to a CSV file with headers and GZIP compression.

# COMMAND ----------

## TASK FOUR - Write df_csv with the following settings
##  - header
##  - compression gzip
##  - overwrite

save_location = "/dbfs/tmp/output_json.json"

## ENTER CODE HERE

# COMMAND ----------

# DBTITLE 1,TASK FOUR ANSWER. Reveal when ready.
save_location = "/dbfs/tmp/output_json.json"

df_csv.write.format("csv") \
    .option("header", "true") \
    .option("compression", "gzip") \
    .mode("overwrite") \
    .save(save_location)

# COMMAND ----------

# MAGIC
# MAGIC
# MAGIC %md
# MAGIC ### Writing Parquet Files
# MAGIC
# MAGIC Write a DataFrame to a Parquet file with Snappy compression.

# COMMAND ----------


## TASK FIVE - Write df_parquet with the following settings
##  - header
##  - compression snappy
##  - append

save_location = "/dbfs/tmp/output_json.json"

## ENTER CODE HERE

# COMMAND ----------

# DBTITLE 1,TASK FIVE ANSWER. Reveal when ready.
save_location = "/dbfs/tmp/output_json.json"

df_parquet.write.format("parquet") \
    .option("compression", "snappy") \
    .mode("append") \
    .save(save_location)