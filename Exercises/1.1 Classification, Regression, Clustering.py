# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Classification, Regression, Clustering*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 1
# MAGIC Pick a dataset to practice classification with. You can use the provided dataset, or the one provided for you.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = #Fill in here

df = (
    #Fill in here
)

target_col = #Fill in here

# COMMAND ----------

#Display the results
#fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2
# MAGIC Create a vector assember, and vector indexer if needed, choose a pyspark ml regression model to go with it

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

# Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset
# MAGIC The cell below shows the splitting of data into a training and test dataset. This is a vital step when building  a machine learninging model since a model needs to be tested on data it hasn't seen before. Typically the test dataset if smaller than the training dataset (20%-30%). Depending on the type of problem, will determine the way it is split. Below uses a random split where the data is shuffled around (removes ordering).

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3
# MAGIC Create a test train split and train the model

# COMMAND ----------

seed = 42
trainDF, testDF = #Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the vector assembler and the model
# MAGIC Below shows the vector assember and the logistic regression model being generated. The vector assembler creates a vectorized column that contains all the features. This needs to done to before passing the dataframe to the classification model. Otherwise and error will be raised. The logistic regression model takes the featuresCol as an argument (the vectorized column containing all the features), as well as which column is the label column. This differs from the Scikit-learn library convention where the label data is passed in a separate argument. 

# COMMAND ----------

columns = #Fill in here
outputCol = #Fill in here
# Put vector assember and routines here

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly. 

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [#Fill in here]
pipeline = Pipeline().setStages(stages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and predicting
# MAGIC Below's command trains the model with the training dataset and uses the test set to generate a prediction, which can be compare with the test data's actual results since the model wouldn't have seen any of the test data in training.

# COMMAND ----------

trained_pipeline = #Fill in here
predictionDF = #Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4
# MAGIC Evalutate the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring the model
# MAGIC The cell below shows how the use can score the model. A couple of metrics are shown below. More information on scoring metrics will be given in another part of the course. 

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = (
    # Fill in here
)
evaluatorac = (
    # Fill in here
)

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
print("Test F1 = %f" % f1)
print("Test Accuracu = %f" % ac)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 5
# MAGIC Create regression model. Load in the data. You can choose the dataset  provided or use your own

# COMMAND ----------

# MAGIC %md 
# MAGIC # Regression

# COMMAND ----------

readPath = # Fill in here

df = (
    #Fill in here
)

label = #Fill in here

# COMMAND ----------

#Display results here
#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 6
# MAGIC Create test train split

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset
# MAGIC The cell below shows the splitting of data into a training and test dataset. This is a vital step when building  a machine learninging model since a model needs to be tested on data it hasn't seen before. Typically the test dataset if smaller than the training dataset (20%-30%). Depending on the type of problem, will determine the way it is split. Below uses a random split where the data is shuffled around (removes ordering).

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit([#Fill in here], #Fill in here)

print(
    "We have %d training examples and %d test examples."
    % (trainDF.count(), testDF.count())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 7
# MAGIC Create vector assembler and train model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the vector assembler and the model
# MAGIC Below shows the vector assember and the linear regression model being generated. The vector assembler creates a vectorized column that contains all the features. This needs to done to before passing the dataframe to the regression model. Otherwise and error will be raised. The LinearRegression model takes the featuresCol as an argument (the vectorized column containing all the features), as well as which column is the label column. This differs from the Scikit-learn library convention where the label data is passed in a separate argument. 

# COMMAND ----------

feats = df.drop(label).columns
outputCol = #Fill in here
# Model and vector assembler. Fill in here

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly.

# COMMAND ----------

from pyspark.ml import Pipeline

stages = #Fill in here
pipeline = Pipeline().setStages([stages])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and predicting
# MAGIC Below's command trains the model with the training dataset and uses the test set to generate a prediction, which can be compare with the test data's actual results since the model wouldn't have seen any of the test data in training. There's two ways of doing this and we will demonstrate both ways:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and predict

# COMMAND ----------

predictionDF = #Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 8
# MAGIC Create a clustering routine. Decide on a clustering dataset. Use the one provided or choose your own. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Clustering

# COMMAND ----------

readPath = #Fill in here

df = (
    #Fill in here
)

# COMMAND ----------

#Display results here
#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 9
# MAGIC Create vector assembler and model. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the vector assembler and the model
# MAGIC Below shows the vector assember and the linear regression model being generated. The vector assembler creates a vectorized column that contains all the features. This needs to done to before passing the dataframe to the regression model. Otherwise and error will be raised. The LinearRegression model takes the featuresCol as an argument (the vectorized column containing all the features), as well as which column is the label column. This differs from the Scikit-learn library convention where the label data is passed in a separate argument.

# COMMAND ----------

dropCol = #Fill in here
feats = df.drop(dropCol).columns
outputCol = #Fill in here
vectorAssembler = VectorAssembler(
    inputCols=feats, outputCol=outputCol, handleInvalid="skip"
)


model = KMeans(featuresCol=outputCol, k=7)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly. 

# COMMAND ----------

from pyspark.ml import Pipeline

stages = #Fill in here
pipeline = Pipeline().setStages(stages)

# COMMAND ----------

trained_pipeline = #Fill in here
predictionsDF = #Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 10
# MAGIC Use silhouette metric to evaluate clustering.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluating the model
# MAGIC The cell below shows how the use can evaluate the model. Unlike regression and clustering, unsupervised methods don't have an explicit way of scoring models. Instead they can be evaluated looking at things like euclidean distance between clusters and visually examining the clustering. 

# COMMAND ----------

evaluator = ClusteringEvaluator(featuresCol="rawFeatures")

silhouette = #Fill in here