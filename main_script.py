
# Step 1: Import Libraries and Load the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
file_path = '/Users/alielsamra/Desktop/ali agenda/CAREER/GITHUB PROJECTS/Smartwatch Data Analysis/dailyActivity_merged.csv' 
data = pd.read_csv(file_path)
print(data.head())
print(data.isnull().sum())
print(data.info())

# Step 2: Preprocessing Data
data["ActivityDate"] = pd.to_datetime(data["ActivityDate"], format="%m/%d/%Y")
print("\nUpdated dataset information:")
print(data.info())
data["TotalMinutes"] = (
    data["VeryActiveMinutes"] +
    data["FairlyActiveMinutes"] +
    data["LightlyActiveMinutes"] +
    data["SedentaryMinutes"]
)
print("\nSample of TotalMinutes column:")
print(data["TotalMinutes"].sample(5))

# Step 3: Descriptive Statistics
print(data.describe())

# Step 4: Relationship Analysis (Calories vs. Steps)
figure = px.scatter(
    data_frame=data,
    x="Calories",
    y="TotalSteps",
    size="VeryActiveMinutes",
    trendline="ols",
    title="Relationship between Calories & Total Steps"
)
figure.show()

# Step 5: Average Active Minutes Analysis
label = ["Very Active Minutes", "Fairly Active Minutes", "Lightly Active Minutes", "Inactive Minutes"]
counts = data[["VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]].mean()
colors = ['gold', 'lightgreen', "pink", "blue"]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Total Active Minutes')
fig.update_traces(
    hoverinfo='label+percent',
    textinfo='value',
    textfont_size=20,
    marker=dict(colors=colors, line=dict(color='black', width=3))
)
fig.show()

# Step 6: Add Day Column and Activity by Day
data["Day"] = data["ActivityDate"].dt.day_name()
print("\nSample of Day column:")
print(data["Day"].head())
fig = go.Figure()
fig.add_trace(go.Bar(x=data["Day"], y=data["VeryActiveMinutes"], name='Very Active', marker_color='purple'))
fig.add_trace(go.Bar(x=data["Day"], y=data["FairlyActiveMinutes"], name='Fairly Active', marker_color='green'))
fig.add_trace(go.Bar(x=data["Day"], y=data["LightlyActiveMinutes"], name='Lightly Active', marker_color='pink'))
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

# Step 7: Inactive Minutes by Day
day_counts = data["Day"].value_counts()
label = day_counts.index
counts = data["SedentaryMinutes"]
colors = ['gold', 'lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Inactive Minutes Daily')
fig.update_traces(
    hoverinfo='label+percent',
    textinfo='value',
    textfont_size=20,
    marker=dict(colors=colors, line=dict(color='black', width=3))
)
fig.show()

# Step 8: Calories by Day
calories_counts = data["Day"].value_counts()
label = calories_counts.index
counts = data["Calories"]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Calories Burned Daily')
fig.update_traces(
    hoverinfo='label+percent',
    textinfo='value',
    textfont_size=20,
    marker=dict(colors=colors, line=dict(color='black', width=3))
)
fig.show()

# Step 9: Add Predictive Analysis 
## First model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

features = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes"]
X = data[features]
y = data["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")

## Second model (improved)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

improved_features = [
    "TotalSteps", 
    "VeryActiveMinutes", 
    "FairlyActiveMinutes", 
    "LightlyActiveMinutes", 
    "SedentaryMinutes", 
    "TotalDistance"
]
X_improved = data[improved_features]
y = data["Calories"]

X_train, X_test, y_train, y_test = train_test_split(X_improved, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_improved = rf_model.predict(X_test)

mse_improved = mean_squared_error(y_test, y_pred_improved)
r2_improved = r2_score(y_test, y_pred_improved)

print(f"Improved Model - Mean Squared Error: {mse_improved}")
print(f"Improved Model - R-Squared: {r2_improved}")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

xgb_model = XGBRegressor(
    n_estimators=300,    # Number of trees
    learning_rate=0.1,   # Step size for optimization
    max_depth=6,         # Maximum depth of each tree
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Model - Mean Squared Error: {mse_xgb}")
print(f"XGBoost Model - R-Squared: {r2_xgb}")

data["Intensity"] = (
    3 * data["VeryActiveMinutes"] + 
    2 * data["FairlyActiveMinutes"] + 
    1 * data["LightlyActiveMinutes"]
)  # Weighting active minutes by intensity

data["ActiveToInactiveRatio"] = (
    (data["VeryActiveMinutes"] + data["FairlyActiveMinutes"] + data["LightlyActiveMinutes"]) /
    (data["SedentaryMinutes"] + 1)  # Adding 1 to avoid division by zero
)
improved_features = [
    "TotalSteps", 
    "VeryActiveMinutes", 
    "FairlyActiveMinutes", 
    "LightlyActiveMinutes", 
    "SedentaryMinutes", 
    "TotalDistance", 
    "Intensity", 
    "ActiveToInactiveRatio"
]
X_improved = data[improved_features]
y = data["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X_improved, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(
    n_estimators=300,    # Number of trees
    learning_rate=0.1,   # Step size for optimization
    max_depth=6,         # Maximum depth of each tree
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate
y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"Enhanced XGBoost Model - Mean Squared Error: {mse_xgb}")
print(f"Enhanced XGBoost Model - R-Squared: {r2_xgb}")

## Third model (last and best model)

import matplotlib.pyplot as plt
import numpy as np
feature_importance = xgb_model.feature_importances_
features = improved_features

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 6, 9],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring="r2",
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1
)
xgb_grid.fit(X_train, y_train)

best_params = xgb_grid.best_params_
best_score = xgb_grid.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best R-Squared from GridSearchCV: {best_score}")

xgb_best = XGBRegressor(**best_params, random_state=42)
xgb_best.fit(X_train, y_train)

y_pred_best = xgb_best.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"Final Model - Mean Squared Error: {mse_best}")
print(f"Final Model - R-Squared: {r2_best}")

import joblib
joblib.dump(xgb_best, "final_xgboost_calorie_predictor.pkl")
print("Model saved as 'final_xgboost_calorie_predictor.pkl'")


