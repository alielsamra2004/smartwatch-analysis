# Smartwatch Data Analysis and Predictive Modeling

This project analyzes smartwatch fitness data, generates visualizations, and builds a predictive model to estimate calories burned. It demonstrates various machine learning techniques using Python libraries.

## Features
- Data preprocessing and exploration.
- Visualizations:
  - Trends in active and inactive minutes.
  - Daily calorie burn and activity patterns.
- Predictive modeling:
  - Linear Regression, Random Forest, and XGBoost.
  - Hyperparameter tuning with GridSearchCV.

## Results
- Final Model:
  - **Mean Squared Error**: 162,320.75
  - **R-Squared**: 0.6566
- Feature importance analysis highlights that **VeryActiveMinutes** and **TotalDistance** are the most influential predictors.

## Repository Structure
## How to Use
1. Clone the repository:
git clone
2. Install dependencies:
pip install -r requirements.txt
3. Run the Jupyter Notebook (`analysis.ipynb`) or the Python script (`main_script.py`) for analysis and modeling.

## Requirements
- Python 3.7+
- Required libraries:
pandas
numpy
matplotlib
plotly
scikit-learn
xgboost
joblib

## License
MIT
