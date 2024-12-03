# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Scaling, Encoding and Imputing*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 1
# MAGIC Load into a dataset you can scale, encode or impute. You can use one provided or choose your own. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = #Fill in here

df = (
    #Fill in here
)

label = #Fill in here

# COMMAND ----------

#Display the results
# Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2
# MAGIC Load in routines and model for use.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.feature import (
    #Fill in here
)
from pyspark.ml.#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit(#Fill in here
)
#Fill in here
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3
# MAGIC Setup scaling and encoding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline with an Scaling and OneHotEncoder
# MAGIC The cell below used the string indexer, one hot encoder, vector assembler and standard scalar. These can then be used within a pipeline (along with model), to create a model. Please note that the standard scalar goes last in the chain. 

# COMMAND ----------

cat_feats = [
    #Fill in here
]
num_feats = [
    #Fill in here
]



inputCols = #Fill in here. Use categorical feature list and label column
outputCols = #Fill in here. Use out_cats list and add update label column
#....
model = #Fill in here


# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4
# MAGIC Setup pipeline

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly. 

# COMMAND ----------

from pyspark.ml import Pipeline

#Fill in here

# COMMAND ----------

predictionDF = # Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 5
# MAGIC Evaluate model

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 6
# MAGIC Create imputation routine

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imputing
# MAGIC Next an imputer can be created to be added to the pipeline. The imputer normally goes first within the pipeline. The impute strategy should be set. Select the appropriate strategy out of mean, median, mode. 

# COMMAND ----------

from pyspark.ml.feature import Imputer

# COMMAND ----------

imp_cols = [
    #Fill in here. Columns you would like to impute
]

inputCols = #Fill in here
outputCols = #Fill in here
imputer = Imputer(inputCols=inputCols, outputCols=outputCols)
imputer.setStrategy(#Fill in here)

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [#Fill in here]
pipeline = Pipeline().setStages(stages)

# COMMAND ----------

predictionDF = # Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring with imputing
# MAGIC Test the scoring again to see if imputing has improved the score. 

# COMMAND ----------

#Fill in here


# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the to Pandas Dataframes
# MAGIC Here we are converting to Pandas Dataframes and creating training and test sets. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 7
# MAGIC Use KNN with scikit-learn

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore KNN imputing and Simple imputing with Scikit-learn
# MAGIC Below is an example of two pipelines used to create two versions of imputing with Scikit-learn. One with imputing, one without imputing. Here you can see the difference between the two imputing methods. A column transformer is used within the pipelines to convert the Pandas dataframes to the transformed version of the data needed for the models. 

# COMMAND ----------

#Fill in here

# COMMAND ----------

#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC Both scores can be compared, one with simple imutation, one with kNN imputation. 

# COMMAND ----------

#Fill in here
print("Score with Simple imputation {:.2f}".format(score))

# COMMAND ----------

#Fill in here
print("Score with kNN imputation {:.2f}".format(score))