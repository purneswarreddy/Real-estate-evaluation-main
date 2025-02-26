import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
dataset_path = "austinHousingData.csv"
df = pd.read_csv(dataset_path)

# Define target variable
target_column = "latestPrice"

# Drop unnecessary columns
drop_columns = ["zpid", "city", "streetAddress", "description", "homeImage", "latestPriceSource", 
                "latest_saledate", "latest_salemonth", "latest_saleyear"]
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# Handle outliers
df = df[df['latitude'] >= 30.12]
df = df[df['numOfBathrooms'] > 0]
df = df[df['numOfBedrooms'] > 0]
df = df[df[target_column] <= 3000000]
df = df[df[target_column] > 75000]
df = df[df['zipcode'] != 78734]

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Feature Selection using RandomForest
def feature_selection(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importance = model.feature_importances_
    selected_features = X.columns[np.argsort(importance)[-30:]]  # Selecting top 30 features
    return X[selected_features]

# Split dataset into features and target variable
X = df.drop(columns=[target_column])
X = feature_selection(X, df[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning function
def tune_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    return grid_search.best_estimator_

# Define hyperparameter grids
xgb_params = {'n_estimators': [100, 300], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1], 'subsample': [0.8, 1.0]}
rf_params = {'n_estimators': [100, 300], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
gb_params = {'n_estimators': [100, 300], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}

# Train optimized models
models = {
    "XGBoost": tune_model(xgb.XGBRegressor(objective="reg:squarederror", random_state=42), xgb_params),
    "Random Forest": tune_model(RandomForestRegressor(random_state=42), rf_params),
    "Gradient Boosting": tune_model(GradientBoostingRegressor(random_state=42), gb_params)
}

# Stacking Regressor
stacked_model = StackingRegressor(
    estimators=[("XGBoost", models["XGBoost"]), ("Random Forest", models["Random Forest"]), ("Gradient Boosting", models["Gradient Boosting"])],
    final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42)
)
stacked_model.fit(X_train_scaled, y_train)

# Evaluate models
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R² Score": r2_score(y_test, y_pred)
    }

# Evaluate Stacking Model
y_pred_stacked = stacked_model.predict(X_test_scaled)
results["Stacked Model"] = {
    "MAE": mean_absolute_error(y_test, y_pred_stacked),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_stacked)),
    "R² Score": r2_score(y_test, y_pred_stacked)
}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

# Display results
print(results_df)
