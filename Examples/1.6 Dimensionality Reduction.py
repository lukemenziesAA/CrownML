# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Dimensionality Reduction*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * How to use PCA for a model in PySpark and Scikit-learn
# MAGIC * How to use RFE impute for a model in Scikit-learn

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

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Splitting the dataset into testing and training

# COMMAND ----------

label = "Attrition_Flag"
seed = 42
drop_cols = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]
X = df.drop(*(drop_cols + [label]))
y = df.select(label)
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=seed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing the libraries needed for PCA
# MAGIC Pyspark has dimensional reduction routines such as PCA which aim to reduce the dimensions of the vectorised dataset. It does this by rotating the vectors around the datapoints that best fits the data whilst keeping orthogonal datapoints. 

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer, PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

# COMMAND ----------

# MAGIC %md
# MAGIC As before, here we need to vectoise the data using the 'vectorAssembler'. Additionally, we need to use 'OneHotEncoder' and 'stringIndexer' to convert he text columns into categorical columns. The PCA object is then defined by passing the vectors in and outputting the reduced vector size. Two models are defined here, one with PCA and one without PCA. 

# COMMAND ----------

cat_data = [i.name for i in X.schema if isinstance(i.dataType, StringType)]
num_data = [i.name for i in X.schema if i.name not in cat_data]
cat_feats_num = [i + "_cat_num" for i in cat_data]
cat_feats = [i + "_cat" for i in cat_data]
vec_feats = num_data + cat_feats
stringIngexer = StringIndexer(
    inputCols=cat_data + [label], outputCols=cat_feats_num + [label + "_num"]
)
ohe = OneHotEncoder(inputCols=cat_feats_num, outputCols=cat_feats)
vectorAssembler = VectorAssembler(
    inputCols=vec_feats, outputCol="rawFeatures", handleInvalid="skip"
)
pca = PCA(k=25, inputCol="rawFeatures", outputCol="features")
model = LogisticRegression(
    featuresCol="rawFeatures", labelCol=label + "_num", maxIter=100
)
model_pca = LogisticRegression(
    featuresCol="features", labelCol=label + "_num", maxIter=100
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup pipelines
# MAGIC There are two pipelines setup. One with and one without PCA. This is so we can compare both results. Pipelines will be covered in a separate section

# COMMAND ----------

pipeline_pca = Pipeline().setStages(
    [stringIngexer, ohe, vectorAssembler, pca, model_pca]
)
pipeline = Pipeline().setStages([stringIngexer, ohe, vectorAssembler, model])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create predictions for both pipelines
# MAGIC Training below gives predictions for both models with PCA and without PCA. Notice that hte training time is dramatically reduced when using PCA due to it being a slimmed down model. 

# COMMAND ----------

predictionDF = pipeline.fit(trainDF).transform(testDF)

# COMMAND ----------

predictionDF_pca = pipeline_pca.fit(trainDF).transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring both pipelines
# MAGIC Below there are two evaluations for the pipelines. One with and one without PCA. You can see that below there isn't a large difference in scoring for both pipeline. The training time for the pipeline model with PCA is noticeably reduced, despite having an additional step. The PCA pipeline only reduced the size down the 25 features from 32 features. In other datasets there may be many obsolete columns that a model may not want to be trained on. Although it doesn't normally lead to better performance, it stream lines the models and also improves training time, which allows for things like grid searches to be able to be performed more in depth for similar compuatational time. 

# COMMAND ----------

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

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
rocauc = evaluatorrocauc.evaluate(predictionDF)
PRauc = evaluatorPRauc.evaluate(predictionDF)
precision = evaluatorPrecision.evaluate(predictionDF)
recall = evaluatorRecall.evaluate(predictionDF)

print("Without PCA")
print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

f1 = evaluatorf1.evaluate(predictionDF_pca)
ac = evaluatorac.evaluate(predictionDF_pca)
rocauc = evaluatorrocauc.evaluate(predictionDF_pca)
PRauc = evaluatorPRauc.evaluate(predictionDF_pca)
precision = evaluatorPrecision.evaluate(predictionDF_pca)
recall = evaluatorRecall.evaluate(predictionDF_pca)

print("With PCA")
print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Scikit-learn's wrapper methods for dimensional reduction
# MAGIC Although PCA is a good technique that can be done using PysPark and Scikit-learn, it isn't a technique that is good for interpreting a model since you lose the original structured columns. Another way you can perform dimensional reduction without losing the interpretability of the features is to use wrapper methods to perform feature elimination. The one below is called Recursive Feature Elimination (RFE). It works by evaltuating the model at every removal of features (looking at feature importance rankings at each step). It does mean the model needs to be able to output feature importance rankings at each step. However, most Scikit-learn models can do this. To improve the elimination further, it can be combined with cross-validation to improve the rankings. This routine is called RFECV. 

# COMMAND ----------

# MAGIC %md
# MAGIC The dataframe has to be converted into a Pandas dataframe, in order to be used with Scikit-learn. 

# COMMAND ----------

df_pandas = df.toPandas()
X = df_pandas.drop(drop_cols + [label], axis=1)
y = df_pandas[label]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the libraries needed

# COMMAND ----------

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

# COMMAND ----------

# MAGIC %md
# MAGIC Split the data into a test and train dataset

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=seed
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make F1 scorer
# MAGIC Scikit-learn makes an F1 scorer since it needs to pass in the positive label. The object is then passed the RFE routine. Otherwise an error will be raised. 

# COMMAND ----------

from sklearn.metrics import f1_score, make_scorer

f1 = make_scorer(f1_score, pos_label="Attrited Customer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create pipeline for RFECV
# MAGIC Similar to PySpark, Scikit-learn has the means of using pipelines to ochestrate the steps. Along with a pipeline, Scikit-learn can use something called a column transformer to generate transformations for numerical and categorical columns. Here 'OneHotEncoder' has been used on categorical data. The numberical values have been scaled between 0 and 1 using the 'MinMaxScalar' routine. There is also a pipeline produced for just the model for comparison. 

# COMMAND ----------

transformer = ColumnTransformer(
    [("cat", OneHotEncoder(), cat_data), ("num", MinMaxScaler(), num_data)]
)
pipeline = Pipeline(
    [
        ("trans", transformer),
        ("RFE", RFECV(LogisticRegression(max_iter=1000), cv=5, scoring=f1)),
    ]
)
pipeline_model = Pipeline(
    [("trans", transformer), ("model", LogisticRegression(max_iter=1000))]
)

# COMMAND ----------

# MAGIC %md
# MAGIC The pipelines are then run using '.fit' and precictions are made using '.predict'

# COMMAND ----------

pipeline.fit(X_train, y_train)
pipeline_model.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
prediction_no_rfe = pipeline_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Here you can navigate through the trained pipelines to get to the step/component of interest. Here you can get the information desired out. Here it has found out which columns were removed using RFECV and what features were left. 

# COMMAND ----------

rfe = pipeline.named_steps["RFE"]
trans = pipeline.named_steps["trans"]
key_feats = rfe.support_
ohe_cats = trans.transformers_[0][1].get_feature_names(cat_data)
orig = pd.DataFrame(trans.transform(X), columns=list(ohe_cats) + num_data).columns
k_feats = orig[key_feats].to_numpy()
removed = list(set(orig) - set(k_feats))
print(
    "{} feature(s) removed: {}".format(rfe.n_features_in_ - rfe.n_features_, *removed)
)
print("Feature columns are: {}".format(k_feats))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparing the two models (with and without RFECV)
# MAGIC As shown, it doesn't effect the score in this case. But it does provide analysis into what features are relevant to the model. Feature importance can also be extracted from this routine with is useful for model interpretability. One thing to note about RFECV is that can be a time consuming method to use. 

# COMMAND ----------

from sklearn.metrics import (
    classification_report,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# COMMAND ----------

classes = y.unique()
results = classification_report(y_test, prediction, labels=classes, output_dict=True)
results_no_rfe = classification_report(
    y_test, prediction_no_rfe, labels=classes, output_dict=True
)

# COMMAND ----------

pd.DataFrame(results_no_rfe)

# COMMAND ----------

pd.DataFrame(results)

# COMMAND ----------

