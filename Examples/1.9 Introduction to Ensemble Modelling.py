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
# MAGIC ## Split the dataframe into a test and train dataset

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
# MAGIC ## Import libraries

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import (
    GBTClassifier,
    RandomForestClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
)
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline, training and predicting

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

model_RF = RandomForestClassifier(
    featuresCol="rawFeatures", labelCol=label + "_num", maxDepth=15, seed=seed
)

model_GBT = GBTClassifier(
    featuresCol="rawFeatures", labelCol=label + "_num", maxDepth=5, seed=seed
)

model_LR = LogisticRegression(
    featuresCol="rawFeatures", labelCol=label + "_num", maxIter=100
)

model_DT = DecisionTreeClassifier(
    featuresCol="rawFeatures", labelCol=label + "_num", maxDepth=15, seed=seed
)

# COMMAND ----------

pipeline_RF = Pipeline().setStages([stringIngexer, ohe, vectorAssembler, model_RF])
pipeline_GBT = Pipeline().setStages([stringIngexer, ohe, vectorAssembler, model_GBT])
pipeline_LR = Pipeline().setStages([stringIngexer, ohe, vectorAssembler, model_LR])
pipeline_DT = Pipeline().setStages([stringIngexer, ohe, vectorAssembler, model_DT])

# COMMAND ----------

# MAGIC %md
# MAGIC Here two predictions are made. One using the undersampled training set and one using the normal training set. 

# COMMAND ----------

predictionDF_RF = pipeline_RF.fit(trainDF).transform(testDF)
predictionDF_GBT = pipeline_GBT.fit(trainDF).transform(testDF)

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

df_pandas = df.toPandas()
X = df_pandas.drop(drop_cols + [label], axis=1)
y = df_pandas[label]

# COMMAND ----------

from sklearn.ensemble import (
    BaggingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    RandomForestClassifier,
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

model = VotingClassifier(
    estimators=[
        ("svm", SVC()),
        ("dt", DecisionTreeClassifier(max_depth=15, random_state=seed)),
    ]
)
trans = ColumnTransformer(
    [("cat", OneHotEncoder(), cat_data), ("num", MinMaxScaler(), num_data)]
)
X_trans = trans.fit_transform(X)
f1 = make_scorer(f1_score, pos_label="Attrited Customer")

# COMMAND ----------

score = cross_val_score(model, X_trans, y.values.ravel(), scoring=f1, cv=5).mean()

# COMMAND ----------

score_dt = cross_val_score(
    DecisionTreeClassifier(max_depth=15, random_state=seed),
    X_trans,
    y.values.ravel(),
    scoring=f1,
    cv=5,
).mean()
score_lr = cross_val_score(
    LogisticRegression(max_iter=1000), X_trans, y.values.ravel(), scoring=f1, cv=5
).mean()
score_svm = cross_val_score(
    SVC(random_state=seed), X_trans, y.values.ravel(), scoring=f1, cv=5
).mean()

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Logistic Regression F1 Score = {:.2f}".format(score_lr))
print("SVC F1 Score = {:.2f}".format(score_svm))
print("Ensamble F1 Score = {:.2f}".format(score))

# COMMAND ----------

model_bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=15, random_state=seed),
    n_estimators=10,
)
score_bag = cross_val_score(
    model_bag, X_trans, y.values.ravel(), scoring=f1, cv=5
).mean()

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Ensemble F1 Score = {:.2f}".format(score_bag))

# COMMAND ----------

model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, random_state=seed))
score = cross_val_score(model, X_trans, y.values.ravel(), scoring=f1, cv=5).mean()

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Ensamble F1 Score = {:.2f}".format(score))

# COMMAND ----------

model = StackingClassifier(
    estimators=[
        ("svm", SVC(random_state=seed)),
        ("dt", DecisionTreeClassifier(max_depth=15, random_state=seed)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=10,
    passthrough=True,
)
score = cross_val_score(model, X_trans, y.values.ravel(), scoring=f1, cv=5).mean()

# COMMAND ----------

print("Decision Tree F1 Score = {:.2f}".format(score_dt))
print("Ensamble F1 Score = {:.2f}".format(score))