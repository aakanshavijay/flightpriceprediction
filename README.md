# flightpriceprediction

Overview- This project focuses on predicting airline flight prices using a machine learning approach. The goal was to apply data science and machine learning techniques to build a model with strong predictive performance.

Dataset-

Source: Kaggle's Flight Price Dataset
Size: 10,000 records of flights
Features: Airline, Date of Journey, Source, Destination, Route, Duration, Total Stops, etc.
Data Preprocessing-

Data Cleaning:

Tackled null values and changed data types of time and date fields.
Feature Engineering:

Extracted hour and minute from Dep_Time, Arrival_Time, and Duration.
Converted Duration to minutes for uniformity.
Analyzed flight schedules using interactive plots (Plotly, Cufflinks)
Investigated the dependency of price on duration using scatter plots.
Analyzed airline usage on different routes.
Applied one-hot encoding on categorical data.
Performed Target Guided Encoding for airlines based on mean prices.
Outlier Detection and Handling:

Visualized outliers using boxplots and distplots.
Applied IQR approach to handle skewness.
Feature Selection:

Used Mutual Information Selection to determine dependencies between variables.
Model Selection:

Trained multiple models: RandomForestRegressor, DecisionTreeRegressor.
Split data into 75% training and 25% testing.
Made my own evaluation model to find metrics like R², MAE, MSE, RMSE, and MAPE.
Results:

1) RandomForestRegressor: Training Score: 0.9517 R² Score: 0.8086 MAPE: 13.27%

2) DecisionTreeRegressor: Training Score: 0.9666 R² Score: 0.6813 MAPE: 15.52%

Hyperparameter Tuning-

RandomizedSearchCV:

Created a random grid of hyperparameters.
Found the best parameters, estimator, and score.
Technologies Used- These technologies and tools were instrumental in building and evaluating my flight price prediction model-

Programming Language: Python: Core language for data processing, analysis, and machine learning.

Libraries/Frameworks:

Pandas: Data manipulation and analysis.
NumPy: Numerical computations and array handling.
Matplotlib: Data visualization.
Seaborn: Statistical data visualization.
Plotly: Interactive plots and visualizations.
Cufflinks: Simplifying the creation of Plotly charts from Pandas dataframes.
Scikit-learn: Machine learning library for model building, feature selection, encoding, and evaluation.
RandomizedSearchCV: Hyperparameter tuning for machine learning models within Scikit-learn.
Machine Learning Models:

RandomForestRegressor: A model for predicting continuous values using an ensemble of decision trees.
DecisionTreeRegressor: A decision tree model for regression tasks.
Data Processing:

Feature Engineering: Extracting and transforming features (e.g., time, duration).
Feature Encoding: Converting categorical data into numerical format.
Outlier Detection and Handling: Using visualizations and statistical methods to manage skewed data.
Mutual Information Selection: For feature selection based on dependencies.
Evaluation Metrics:

R² Score, MAE, MSE, RMSE, MAPE: Standard evaluation metrics to assess model performance.
Custom Evaluation Function: A self-developed metric to evaluate multiple aspects of model performance. These technologies and tools were instrumental in building and evaluating your flight price prediction model.
How to Execute-

Clone the repository.
Install dependencies: pip install -r requirements.txt
Run the Jupyter notebook to explore and train the model.
Conclusion- This model provides accurate flight price predictions, with the RandomForestRegressor achieving the highest accuracy. Further tuning and more data could potentially improve performance.
