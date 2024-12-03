# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Custom Transformer*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * How to create a custom transformer for a machine learning pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 1
# MAGIC Load into a dataset you can scale, encode or impute. You can use one provided or choose your own. Also load in appropriate libraries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

#Fill in here

# COMMAND ----------

# DBTITLE 0,--i18n-3e08ca45-9a00-4c6a-ac38-169c7e87d9e4
# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2
# MAGIC Perform test train split

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset
# MAGIC The cell below shows the splitting of data into a training and test dataset. This is a vital step when building  a machine learninging model since a model needs to be tested on data it hasn't seen before. Typically the test dataset if smaller than the training dataset (20%-30%). Depending on the type of problem, will determine the way it is split. Below uses a random split where the data is shuffled around (removes ordering).

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3
# MAGIC Create a custom transformer of your choosing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Created Custom Transformer
# MAGIC This is a custom transformer that imports and inherets 'Transformer' into the class. It can be used for machine learning transformation, and passed into machine learning pipelines. It receives a PySpark DataFrame requires manipulation of using PySpark.

# COMMAND ----------

class CustomTransformer(Transformer):

    def __init__():
        #Fill in here
    def _transform(self, df):
        #Fill in here
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create pipeline with transformer
# MAGIC Create pipeline with custom transformer in the pipeline stages.

# COMMAND ----------

#Fill in here
model = Pipeline().setStages([#Fill in here

# COMMAND ----------

pipeline_model = model.fit(trainDF)
predictionDF = pipeline_model.transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4
# MAGIC Score the model that uses the transformer.

# COMMAND ----------

# DBTITLE 0,--i18n-8d5f8c24-ee0b-476e-a250-95ce2d73dd28
# MAGIC %md
# MAGIC ## Scoring the model
# MAGIC The cell below shows how the use can score the model. A couple of metrics are shown below. More information on scoring metrics will be given in another part of the course. 

# COMMAND ----------

#Fill in here