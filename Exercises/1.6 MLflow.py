# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *MLflow*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Exercise: Starting Runs, Logging Parameters, and Registering Models
# MAGIC  
# MAGIC This notebook is designed to test your understanding of MLflow by asking you to complete various tasks related to:
# MAGIC 1. Starting MLflow runs
# MAGIC 2. Logging custom parameters and metrics
# MAGIC 3. Registering a model in the MLflow Model Registry
# MAGIC
# MAGIC Follow the instructions and fill in the code where necessary.
# MAGIC

# COMMAND ----------

# Import necessary libraries
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# COMMAND ----------

# TASK ONE: Load /databricks-datasets/wine-quality/winequality-white.csv
spark_df = "ENTER CODE HERE"

# Convert to a Pandas DataFrame
df = spark_df.toPandas()

# COMMAND ----------

# DBTITLE 1,TASK ONE ANSWER. Reveal when ready.
# Load the dataset as a Spark DataFrame

spark_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("sep", ";").load("/databricks-datasets/wine-quality/winequality-white.csv")

# Convert to a Pandas DataFrame
df = spark_df.toPandas()

# COMMAND ----------

# Prepare the feature matrix `X` and target vector `y`
X = df.drop("quality", axis=1)
y = df["quality"] >= 7  # Binary classification: quality >= 7 is good

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### Step 2: Define Model Training and Logging Process
# MAGIC  
# MAGIC Define a function `train_and_log_model` that trains a model and logs parameters, metrics, and the model itself using MLflow.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - Start an MLflow run.
# MAGIC - Train a `RandomForestClassifier` with given parameters.
# MAGIC - Log the parameters and metrics.
# MAGIC - Log and register the trained model.
# MAGIC

# COMMAND ----------

def train_and_log_model(params):
    with mlflow.start_run(run_name=params["run_name"]):
        # Set up the model
        model = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        
        ## TASK TWO: Log the following custom parameters: n_estimators, dataset_name, accuracy.
        
        
        # TASK THREE: Log the model with the name "random-forest-model"

        
        # TASK FOUR: Register the model with model_name
        model_name = params["model_name"]
        run_id = mlflow.active_run().info.run_id
    
        ## Print results
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model registered as: {model_name}")


# COMMAND ----------

# Define the parameters for the run
params = {
    "run_name": "Random Forest Classifier",
    "n_estimators": 100,
    "model_name": "RandomForestWineQuality"
}

# Train the model and log the details
train_and_log_model(params)

# COMMAND ----------

# DBTITLE 1,TASK TWO, THREE & FOUR ANSWER. Reveal when ready.
def train_and_log_model(params):
    with mlflow.start_run(run_name=params["run_name"]):
        # Set up the model
        model = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("dataset", "wine-quality-white")
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(model, "random-forest-model")
        
        # Register the model
        model_name = params["model_name"]
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/random-forest-model", model_name)
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model registered as: {model_name}")


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### Step 3: Viewing Results
# MAGIC
# MAGIC - Navigate to the MLflow UI to see the logged runs and metrics.
# MAGIC - Check the Model Registry to see the registered model.
# MAGIC
# MAGIC **Instructions:**
# MAGIC - Verify that the parameters and metrics have been logged correctly.
# MAGIC - Verify that the model has been registered in the Model Registry.