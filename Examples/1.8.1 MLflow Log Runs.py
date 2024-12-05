# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *MLflow Log Runs*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md # Log runs to an experiment

# COMMAND ----------

# MAGIC %md ## Types of experiments
# MAGIC There are two types of experiments in MLflow: _notebook_ and _workspace_. 
# MAGIC * A notebook experiment is associated with a specific notebook. Databricks creates a notebook experiment by default when a run is started using `mlflow.start_run()` and there is no active experiment.
# MAGIC * Workspace experiments are not associated with any notebook, and any notebook can log a run to these experiments by using the experiment name or the experiment ID when initiating a run. 
# MAGIC
# MAGIC This notebook creates a Random Forest model on a simple dataset and uses the MLflow Tracking API to log the model and selected model parameters and metrics.

# COMMAND ----------

# Import the dataset from scikit-learn and create the training and test datasets. 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

# MAGIC %md By default, MLflow runs are logged to the notebook experiment, as illustrated in the following code block.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# In this run, neither the experiment_id nor the experiment_name parameter is provided. MLflow automatically creates a notebook experiment and logs runs to it.
# Access these runs using the Experiment sidebar. Click Experiment at the upper right of this screen. 

with mlflow.start_run():
  n_estimators = 100
  max_depth = 6
  max_features = 3
  # Create and train model
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  # Make predictions
  predictions = rf.predict(X_test)
  
  # Log parameters
  mlflow.log_param("num_trees", n_estimators)
  mlflow.log_param("maxdepth", max_depth)
  mlflow.log_param("max_feat", max_features)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
    
  # Log metrics
  mlflow.log_metric("mse", mse)

# COMMAND ----------

# MAGIC %md To log MLflow runs to a workspace experiment, use `mlflow.set_experiment()` as illustrated in the following code block. An alternative is to set the experiment_id parameter in `mlflow.start_run()`; for example, `mlflow.start_run(experiment_id=1234567)`.

# COMMAND ----------

# This run uses mlflow.set_experiment() to specify an experiment in the workspace where runs should be logged. 
# If the experiment specified by experiment_name does not exist in the workspace, MLflow creates it.
# Access these runs using the experiment name in the workspace file tree. 

experiment_name = "/Shared/diabetes_experiment/"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
  n_estimators = 100
  max_depth = 6
  max_features = 3
  # Create and train model
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  # Make predictions
  predictions = rf.predict(X_test)
  
  # Log parameters
  mlflow.log_param("num_trees", n_estimators)
  mlflow.log_param("maxdepth", max_depth)
  mlflow.log_param("max_feat", max_features)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
    
  # Log metrics
  mlflow.log_metric("mse", mse)