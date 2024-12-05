# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Introduction to Classification*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = "dbfs:/FileStore/tables/credit_card_churn_clean2.csv"

df = (
    spark.read.option("header", True)
    .format("csv")
    .option("inferSchema", True)
    .load(readPath)
)

target_col = "Attrition_Flag"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.feature import (
    VectorIndexer,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset
# MAGIC The cell below shows the splitting of data into a training and test dataset. This is a vital step when building  a machine learninging model since a model needs to be tested on data it hasn't seen before. Typically the test dataset if smaller than the training dataset (20%-30%). Depending on the type of problem, will determine the way it is split. Below uses a random split where the data is shuffled around (removes ordering).

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=seed)

print(
    "We have %d training examples and %d test examples."
    % (trainDF.count(), testDF.count())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the vector assembler and the model
# MAGIC Below shows the vector assember and the logistic regression model being generated. The vector assembler creates a vectorized column that contains all the features. This needs to done to before passing the dataframe to the classification model. Otherwise and error will be raised. The logistic regression model takes the featuresCol as an argument (the vectorized column containing all the features), as well as which column is the label column. This differs from the Scikit-learn library convention where the label data is passed in a separate argument. 

# COMMAND ----------

df.display()

# COMMAND ----------

df.columns

# COMMAND ----------

columns = list(set(df.columns) - set([target_col]))
vectorAssembler = VectorAssembler(
    inputCols=columns, outputCol="rawFeatures", handleInvalid="skip"
)
 
model = LogisticRegression(
    featuresCol="rawFeatures", labelCol=target_col, maxIter=20
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly. 

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages([vectorAssembler, model])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and predicting
# MAGIC Below's command trains the model with the training dataset and uses the test set to generate a prediction, which can be compare with the test data's actual results since the model wouldn't have seen any of the test data in training.

# COMMAND ----------

trained_pipeline = pipeline.fit(trainDF)
predictionDF = trained_pipeline.transform(testDF)

# COMMAND ----------

predictionDF.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring the model
# MAGIC The cell below shows how the use can score the model. A couple of metrics are shown below. More information on scoring metrics will be given in another part of the course. 

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricName("f1")
    .setPredictionCol("prediction")
    .setLabelCol(target_col)
)
evaluatorac = (
    MulticlassClassificationEvaluator()
    .setMetricName("accuracy")
    .setPredictionCol("prediction")
    .setLabelCol(target_col)
)

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
print("Test F1 = %f" % f1)
print("Test Accuracu = %f" % ac)

# COMMAND ----------

