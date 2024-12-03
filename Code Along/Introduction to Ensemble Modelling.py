# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Introduction to Ensemble Modelling*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * How to use ensemble methods in PySpark
# MAGIC * How to use bagging, boosting and stacking ensemble methods in Scikit-learn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = # Fill in here

df = (
    # Fill in here
)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset

# COMMAND ----------

label = "Attrition_Flag"
seed = 42
drop_cols = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]
X = # Fill in here
y = # Fill in here
trainDF, testDF = # Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import (
    # Fill in here
)
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline, training and predicting

# COMMAND ----------

cat_data = [i.name for i in X.schema if isinstance(i.dataType, StringType)]
num_data = [i.name for i in X.schema if i.name not in cat_data]
cat_feats_num = # Fill in here
cat_feats = # Fill in here
vec_feats = # Fill in here
stringIngexer = # Fill in here

ohe = # Fill in here

vectorAssembler = # Fill in here

model_RF = # Fill in here

model_GBT = # Fill in here

model_LR = # Fill in here

model_DT = # Fill in here

# COMMAND ----------

pipeline_RF = Pipeline().setStages(# Fill in here)
pipeline_GBT = Pipeline().setStages(# Fill in here)
pipeline_LR = Pipeline().setStages(# Fill in here)
pipeline_DT = Pipeline().setStages(# Fill in here)

# COMMAND ----------

predictionDF_RF = # Fill in here
predictionDF_GBT = # Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring the model in PySpark

# COMMAND ----------

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("f1")
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorac = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("accuracy")
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorPrecision = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("precisionByLabel")
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorRecall = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("recallByLabel")
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorrocauc = (
    BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorPRauc = (
    BinaryClassificationEvaluator()
    .setMetricName("areaUnderPR")
    .setRawPredictionCol("prediction")
    .setLabelCol(label + "_num")
)

# COMMAND ----------

f1 = evaluatorf1.evaluate(predictionDF_GBT)
ac = evaluatorac.evaluate(predictionDF_GBT)
rocauc = evaluatorrocauc.evaluate(predictionDF_GBT)
PRauc = evaluatorPRauc.evaluate(predictionDF_GBT)
precision = evaluatorPrecision.evaluate(predictionDF_GBT)
recall = evaluatorRecall.evaluate(predictionDF_GBT)

print("GBT Classifier")
print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

f1 = evaluatorf1.evaluate(predictionDF_RF)
ac = evaluatorac.evaluate(predictionDF_RF)
rocauc = evaluatorrocauc.evaluate(predictionDF_RF)
PRauc = evaluatorPRauc.evaluate(predictionDF_RF)
precision = evaluatorPrecision.evaluate(predictionDF_RF)
recall = evaluatorRecall.evaluate(predictionDF_RF)

print("Random Forest Classifier")
print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

predictionDF_LR = pipeline_LR.fit(trainDF).transform(testDF)
predictionDF_DT = pipeline_DT.fit(trainDF).transform(testDF)

# COMMAND ----------

f1 = evaluatorf1.evaluate(predictionDF_DT)
ac = evaluatorac.evaluate(predictionDF_DT)
rocauc = evaluatorrocauc.evaluate(predictionDF_DT)
PRauc = evaluatorPRauc.evaluate(predictionDF_DT)
precision = evaluatorPrecision.evaluate(predictionDF_DT)
recall = evaluatorRecall.evaluate(predictionDF_DT)

print("Decision Tree Classifier")
print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

f1 = evaluatorf1.evaluate(predictionDF_LR)
ac = evaluatorac.evaluate(predictionDF_LR)
rocauc = evaluatorrocauc.evaluate(predictionDF_LR)
PRauc = evaluatorPRauc.evaluate(predictionDF_LR)
precision = evaluatorPrecision.evaluate(predictionDF_LR)
recall = evaluatorRecall.evaluate(predictionDF_LR)

print("Logistic Regression Classifier")
print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

df_pandas = # Fill in here
X = # Fill in here
y = # Fill in here

# COMMAND ----------

from sklearn.ensemble import (
    # Fill in here
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import make_scorer, f1_score

# COMMAND ----------

model = VotingClassifier(# Fill in here
)
trans = ColumnTransformer(
    [("cat", OneHotEncoder(), cat_data), ("num", MinMaxScaler(), num_data)]
)
X_trans = trans.fit_transform(X)
f1 = make_scorer(# Fill in here)

# COMMAND ----------

score = # Fill in here

# COMMAND ----------

score_dt = cross_val_score(# Fill in here
score_lr = cross_val_score(# Fill in here
score_svm = cross_val_score(# Fill in here

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Logistic Regression F1 Score = {:.2f}".format(score_lr))
print("SVC F1 Score = {:.2f}".format(score_svm))
print("Ensamble F1 Score = {:.2f}".format(score))

# COMMAND ----------

model_bag = BaggingClassifier(# Fill in here
score_bag = cross_val_score(# Fill in here

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Ensemble F1 Score = {:.2f}".format(score_bag))

# COMMAND ----------

model = AdaBoostClassifier(# Fill in here)
score = cross_val_score(# Fill in here

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Ensamble F1 Score = {:.2f}".format(score))

# COMMAND ----------

model = StackingClassifier(# Fill in here)
score = cross_val_score(# Fill in here

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Ensamble F1 Score = {:.2f}".format(score))