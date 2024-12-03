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
# MAGIC * How to optimise tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1
# MAGIC Load in Libraries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2
# MAGIC Load in data. Choose your own dataset or pick one provided

# COMMAND ----------

# DBTITLE 0,--i18n-3e08ca45-9a00-4c6a-ac38-169c7e87d9e4
# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3
# MAGIC Perform aggregations. Create aggregations on using group by. Create aggregation using the agg function, one with a single aggregation, on with a new aggregation. 

# COMMAND ----------

# MAGIC %md
# MAGIC Various aggregate methods are available on the grouped data object

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4
# MAGIC Apply a filter to column 

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5
# MAGIC Create Window function to minus the average of a value within that window from another one. 

# COMMAND ----------

# MAGIC %md
# MAGIC Window functions can be used to perform operations on a window relative to each row. 

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 6
# MAGIC Perform optimisations on a tables saved to a schema

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 7
# MAGIC Show visualisations and add them to a dashboard

# COMMAND ----------

#Fill in here