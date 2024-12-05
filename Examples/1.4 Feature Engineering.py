# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Feature Engineering*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * How to perform group by with aggregations
# MAGIC * How to create window functions
# MAGIC * How to use when clauses
# MAGIC * How to extract datetime feature
# MAGIC * How to optimise tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.sql.functions import log, col
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.sql.functions import avg, approx_count_distinct
from pyspark.sql.window import Window
from pyspark.sql.functions import mean, sum, count, dayofmonth, month, year, when, udf
import re

# COMMAND ----------

# DBTITLE 0,--i18n-3e08ca45-9a00-4c6a-ac38-169c7e87d9e4
# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

df = spark.table("catalog.schema.table")

# COMMAND ----------

readPath = "dbfs:/FileStore/tables/credit_card_churn.csv"

drop_cols = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

df = (
    spark.read.option("header", True)
    .format("csv")
    .option("inferSchema", True)
    .load(readPath)
).drop(*drop_cols)

target_col = "Attrition_Flag"

# COMMAND ----------

taxis = (
    spark.read.format("parquet")
    .load("dbfs:/FileStore/tables/taxi_snippet.parquet")
    .withColumn("pickup_datetime", col("pickup_datetime").cast("timestamp"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Various aggregate methods are available on the grouped data object

# COMMAND ----------

maritalCountDF = df.groupBy("Marital_Status").count()
display(maritalCountDF)

# COMMAND ----------

avgMaritalStatusAgeDF = df.groupBy("Marital_Status").avg("Customer_Age")
display(avgMaritalStatusAgeDF)

# COMMAND ----------

genderCustomerAgeSumDependentDF = df.groupBy("Gender", "Customer_Age").sum(
    "Dependent_count"
)
display(genderCustomerAgeSumDependentDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the grouped data method `agg` to apply built-in aggregate functions
# MAGIC This allows you to apply other transformations on the resulting columns, such as `alias`. You can also apply multiple aggregation methods.

# COMMAND ----------

statusAggregatesDF = df.groupBy("Marital_Status").agg(
    avg("Customer_Age").alias("avg_customer_age"),
    count("Customer_Age").alias("tot_customer_age"),
)

# COMMAND ----------

display(statusAggregatesDF)

# COMMAND ----------

df.agg(mean("Customer_Age")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Apply filters to filter on specific categories

# COMMAND ----------

df.filter(col("Education_Level") == "college")

# COMMAND ----------

# MAGIC %md
# MAGIC Window functions can be used to perform operations on a window relative to each row. 

# COMMAND ----------

windowSpec = Window.partitionBy("Education_Level")
df = df.withColumn(
    "Age_minus_AvgAge", col("Customer_Age") - avg("Customer_Age").over(windowSpec)
)

display(df)

# COMMAND ----------

windowSpec = Window.orderBy("pickup_datetime")
spark_df = taxis.withColumn("cum_sum", count("vendor_id").over(windowSpec))

display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Datetime features for ML can be used to extract further features such and day of month, month and year using the PySpark libary.

# COMMAND ----------



# COMMAND ----------

spark_df = (
    spark_df.withColumn("year", year("pickup_datetime"))
    .withColumn("month", month("pickup_datetime"))
    .withColumn("day", dayofmonth("pickup_datetime"))
)

display(spark_df)

# COMMAND ----------

spark_df.rdd.getNumPartitions()

# COMMAND ----------

repartitionedDF = spark_df.repartition(2)

# COMMAND ----------

# MAGIC %md
# MAGIC + Aim for 2-4x the number of partitions compared to the number of cores available in the cluster. This helps ensure full utilization of cluster resources.
# MAGIC + Each task (partition) should take at least 100ms to execute. If tasks are taking less time, the partitioned data may be too small, leading to inefficiencies in task distribution.

# COMMAND ----------

spark_df.write.saveAsTable("taxi_data.taxi_snippet")

# COMMAND ----------

# MAGIC %md
# MAGIC Optimise can be used to optimise the tables. 

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE taxi_data.taxi_snippet

# COMMAND ----------

taxis = taxis.withColumn(
    "trip_type", when(col("trip_distance") > 5, "long").otherwise("short")
)
taxis.select("trip_distance", "trip_type").show(5)

# COMMAND ----------

# Convert "rate_code_id" from string to integer
taxis = taxis.withColumn("rate_code_id", col("rate_code_id").cast("integer"))
taxis.printSchema()

# COMMAND ----------

