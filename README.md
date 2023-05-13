# Project Description

* This repository contains my own price prediction model for the "House Prices - Advanced Regression Techniques" dataset on Kaggle.

# Model Building

* The model was built using the LightGBM Regressor based on the analysis conducted in the "research.ipynb" notebook.

* The model was trained only on the "train.csv" dataset, and the "test.csv" dataset was not used during the model building or analysis stage.

* Therefore, the main goal of the model building was to make generalizable predictions for the unseen data in the "test.csv" dataset, even during the research  phase.

* The model showed its best performance in making predictions for the "test.csv" dataset.

# Prediction Pipeline

* The "prediction_pipeline.py" file makes predictions for the test set when run in the terminal.

* The results are saved as an .xlsx file to the path you specify.