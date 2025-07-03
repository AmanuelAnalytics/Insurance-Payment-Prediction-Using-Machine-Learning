# 🚀 Insurance Payment Prediction - Full Pipeline with No Data Leakage (Model-Specific Scaling)

# Install if needed
# !pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 1️⃣ Load Data
# ===============================
df = pd.read_excel('insurance_information .xlsx')
df = df.drop('Unnamed: 3', axis=1)
df.columns = ['Insurance Provider', 'Service Type', 'Miles', 'Payment Amount']
df.dropna(inplace=True)

# ===============================
# 2️⃣ Feature Engineering (Safe)
# ===============================
df['Miles Category'] = pd.cut(df['Miles'],
                              bins=[0, 5, 15, 30, df['Miles'].max() + 1],
                              labels=['Very Short', 'Short', 'Medium', 'Long'])
df['High Mileage'] = (df['Miles'] > 20).astype(int)
service_map = {
    'Wheelchair': 1,
    'Basic Life Support NonEmergency': 2,
    'Advanced Life Support NonEmergency': 3
}
df['Service Complexity'] = df['Service Type'].map(service_map).fillna(2)

# ===============================
# 3️⃣ Train-Test Split
# ===============================
X = df[['Insurance Provider', 'Service Type', 'Miles', 'Miles Category', 'High Mileage', 'Service Complexity']]
y = df['Payment Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 4️⃣ Base Rate Without Leakage
# ===============================
base_rate_map = pd.DataFrame({'Base Rate': y_train}).groupby(
    [X_train['Insurance Provider'], X_train['Service Type']]
).mean().reset_index()

X_train = X_train.merge(base_rate_map, on=['Insurance Provider', 'Service Type'], how='left')
X_test = X_test.merge(base_rate_map, on=['Insurance Provider', 'Service Type'], how='left')
X_test['Base Rate'] = X_test['Base Rate'].fillna(y_train.mean())

# ===============================
# 5️⃣ Preprocessing
# ===============================
categorical_features = ['Insurance Provider', 'Service Type', 'Miles Category']
numeric_features = ['Miles', 'High Mileage', 'Service Complexity', 'Base Rate']

# For Linear Models: With Scaling
preprocessor_scaled = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

# For Tree-Based Models: No Scaling
preprocessor_unscaled = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# ===============================
# 6️⃣ Evaluation Function
# ===============================
def evaluate(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{model_name} Performance:")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

# ===============================
# 7️⃣ Ridge and Lasso (Scaled)
# ===============================
ridge_model = Pipeline([
    ('preprocessor', preprocessor_scaled),
    ('regressor', Ridge(alpha=1.0))
])
ridge_model.fit(X_train, y_train)
evaluate(y_test, ridge_model.predict(X_test), "Ridge Regression")

lasso_model = Pipeline([
    ('preprocessor', preprocessor_scaled),
    ('regressor', Lasso(alpha=1.0))
])
lasso_model.fit(X_train, y_train)
evaluate(y_test, lasso_model.predict(X_test), "Lasso Regression")

# ===============================
# 8️⃣ Random Forest (Unscaled)
# ===============================
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('regressor', RandomForestRegressor(random_state=42))
])
rf_params = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [5, 10, None]
}
grid_rf = GridSearchCV(rf_pipeline, rf_params, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
evaluate(y_test, best_rf.predict(X_test), "Random Forest (Tuned)")

# ===============================
# 9️⃣ XGBoost (Unscaled)
# ===============================
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])
xgb_params = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.05, 0.1, 0.2]
}
grid_xgb = GridSearchCV(xgb_pipeline, xgb_params, cv=5, n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
evaluate(y_test, best_xgb.predict(X_test), "XGBoost (Tuned)")

# ===============================
# 🔟 Stacking Ensemble
# ===============================
stacking_model = StackingRegressor(
    estimators=[
        ('ridge', ridge_model),
        ('rf', best_rf),
        ('xgb', best_xgb)
    ],
    final_estimator=LinearRegression()
)
stacking_model.fit(X_train, y_train)
evaluate(y_test, stacking_model.predict(X_test), "Stacking Ensemble")
