# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Nested Runs in MLflow*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# COMMAND ----------

# Load the dataset
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("sep", ";").load("/databricks-datasets/wine-quality/winequality-white.csv")
df = df.toPandas()

# COMMAND ----------

display(df)

# COMMAND ----------

# Prepare the data
X = df.drop("quality", axis=1)
y = df["quality"] >= 7  # Let's treat quality >= 7 as good quality (binary classification)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# Start a parent run
with mlflow.start_run(run_name="Parent Run") as parent_run:
    # Log a parameter in the parent run
    mlflow.log_param("dataset", "wine-quality-white")
    
    # Start a child run for the Random Forest model
    with mlflow.start_run(run_name="Random Forest Run", nested=True) as rf_run:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Log model and metrics
        mlflow.sklearn.log_model(rf, "random-forest-model")
        mlflow.log_metric("accuracy", rf_accuracy)
    
    # Start a child run for the Logistic Regression model
    with mlflow.start_run(run_name="Logistic Regression Run", nested=True) as lr_run:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        # Log model and metrics
        mlflow.sklearn.log_model(lr, "logistic-regression-model")
        mlflow.log_metric("accuracy", lr_accuracy)
    
    # Optionally log the metrics of the child runs in the parent run
    mlflow.log_metric("rf_accuracy", rf_accuracy)
    mlflow.log_metric("lr_accuracy", lr_accuracy)