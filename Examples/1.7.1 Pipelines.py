# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Pipelines*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * How to build a pipeline in PySpark and Scikit-learn
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The routine below demonstrates an example of when data leakage occurs without the use of a pipeline. It trains a model on completely random data (no patterns). Clearly this would give a poor score, however, after a tranformation, the score gives a unrealistically high value when not put into a pipeline.

# COMMAND ----------

from numpy.random import RandomState
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Model trained and scored not in a pipeline

# COMMAND ----------

rnd = RandomState(seed=0)
X = rnd.random(size=(100, 10000))
y = rnd.random(size=(100,))


select = SelectPercentile(score_func=f_regression,
                          percentile=5)
select.fit(X, y)
X_selected = select.transform(X)
print(f"Reduced the dimension down from {X.shape[1]} to {X_selected.shape[1]}")

score = cross_val_score(Ridge(), X_selected, y).mean()
print("The R2 score is giving a decieving value of {:.2f}".format(score))

# COMMAND ----------

# MAGIC %md
# MAGIC Model trained and scored using a pipeline

# COMMAND ----------

pipe = Pipeline([("select", SelectPercentile(score_func=f_regression,
                                             percentile=5)),
                 ("ridge", Ridge())])

score = cross_val_score(pipe, X, y).mean()
print("Putting the feature selection and model into a pipeline removes the potential to missrepresent accuracy of the model: new score = {:.2f}".format(score))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = "dbfs:/FileStore/tables/credit_card_churn.csv"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .format("csv")
    .load(readPath)
)

label = "Attrition_Flag"

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC caching the dataframe improves subsequent load times. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset
# MAGIC The cell below shows the splitting of data into a training and test dataset. This is a vital step when building  a machine learninging model since a model needs to be tested on data it hasn't seen before. Typically the test dataset if smaller than the training dataset (20%-30%). Depending on the type of problem, will determine the way it is split. Below uses a random split where the data is shuffled around (removes ordering).

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=42)

print("We have %d training examples and %d test examples." % (trainDF.count(), testDF.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the vector assembler and the model
# MAGIC Below shows the vector assember and the logistic regression model being generated as well as the putting the string columns in categorical form. Machine learning models do not understand string columns as they are. They need to be transformed into categorical form where each unique category in the column has a unique column with a binary value. Prior to this, the routine 'stringIndexer' is needed to asign an integer values to a string category. This is so it can be converted into a category. The vector assembler creates a vectorized column that contains all the features. This needs to done to before passing the dataframe to the classification model. Otherwise and error will be raised. The logistic regression model takes the featuresCol as an argument (the vectorized column containing all the features), as well as which column is the label column. This differs from the Scikit-learn library convention where the label data is passed in a separate argument.

# COMMAND ----------

from pyspark.sql.types import DoubleType, IntegerType

# COMMAND ----------

cat_feats = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
num_feats = [
    "Customer_Age",
    "Dependent_count",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
] 

out_cats = [i + "_catv" for i in cat_feats]
f_cats = [i + "_cat" for i in cat_feats]
vec_feats = num_feats + f_cats


stringIndexer = StringIndexer(
    inputCols=cat_feats + ["Attrition_Flag"],
    outputCols=out_cats + ["Attrition_Flag_num"],
)
ohe = OneHotEncoder(inputCols=out_cats, outputCols=f_cats)
vectorAssembler = VectorAssembler(
    inputCols=vec_feats, outputCol="rawFeatures", handleInvalid="skip"
)

model = DecisionTreeClassifier(
    featuresCol="rawFeatures", labelCol="Attrition_Flag_num"
)  # maxIter=100)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model using Spark. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. 

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages([stringIndexer, ohe, vectorAssembler, model])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and predicting
# MAGIC Below's command trains the model with the training dataset and uses the test set to generate a prediction, which can be compare with the test data's actual results since the model wouldn't have seen any of the test data in training. 

# COMMAND ----------

predictionDF = pipeline.fit(trainDF).transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring the model
# MAGIC The cell below shows how the use can score the model.

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = MulticlassClassificationEvaluator().setMetricName("f1").setPredictionCol("prediction").setLabelCol("Attrition_Flag_num")
evaluatorac = MulticlassClassificationEvaluator().setMetricName("accuracy").setPredictionCol("prediction").setLabelCol("Attrition_Flag_num")

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
print("Test F1 = %f" % f1)
print("Test Accuracu = %f" % ac)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scikit-learn Pipelines

# COMMAND ----------

X_train = trainDF.drop("No", label, "rawFeatures").toPandas()
y_train = trainDF.select(label).toPandas()
X_test = testDF.drop("No", label, "rawFeatures").toPandas()
y_test = testDF.select(label).toPandas()

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## ColumnTransformer
# MAGIC Below makes use of a column transformer to transform columns of a specific type within a Pandas Dataframe. Scaling is performed on the numerical values and encoding is performed on the categorical columns. Both have simple imputation implemented. 

# COMMAND ----------

model = LogisticRegression(max_iter=1000)
cat_pipeline = Pipeline([('enc', OneHotEncoder()), ('imp', SimpleImputer(strategy='most_frequent'))])
num_pipeline = Pipeline([('scale', MinMaxScaler()), ('imp', SimpleImputer(strategy='mean'))])
transformer = ColumnTransformer([('cat', cat_pipeline, cat_feats), ('numeric', num_pipeline, num_feats)])
pipeline = Pipeline([('tranform', transformer), ('model', model)])

# COMMAND ----------

# MAGIC %md
# MAGIC Train the model

# COMMAND ----------

pipeline.fit(X_train, y_train.values.ravel())

# COMMAND ----------

# MAGIC %md
# MAGIC Score the model

# COMMAND ----------

ac = pipeline.score(X_test, y_test)
print("Test accuracy = %f" % ac)

# COMMAND ----------

