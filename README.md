#  Insurance Payment Prediction Using Machine Learning

##  Project Objective

The objective of this project is to build a machine learning model that accurately predicts **insurance reimbursement amounts** for medical transportation services. The model is based on three main features:

- Insurance Provider  
- Service Type  
- Miles Transported  

The predictive tool helps transportation companies estimate reimbursement amounts with high reliability, enabling **better billing, budgeting, and profitability analysis**.

---

##  Dataset Overview

The dataset includes real-world or synthetic historical records with the following columns:

- `Insurance Provider`: The company paying the reimbursement  
- `Service Type`: Type of transport (Wheelchair, Basic Life Support, etc.)  
- `Miles`: Distance of the trip  
- `Payment Amount`: The amount reimbursed (target variable)

> **Note**: The original dataset is not included due to usage restrictions. A synthetic or anonymized version can be requested for demo purposes.

---

##  Methodology

###  Feature Engineering

To maximize model performance, the following features were engineered:

- **Miles Category**: Groups trips into `Very Short`, `Short`, `Medium`, and `Long`  
- **High Mileage Flag**: Indicates whether a trip exceeded 20 miles  
- **Service Complexity**: Numeric mapping of transport complexity  
- **Base Rate**: Mean historical reimbursement per Insurance + Service Type (calculated **without data leakage**)

These engineered features added predictive power beyond the three original columns.

---

###  Model-Specific Preprocessing Strategy

A tailored approach was used to preprocess inputs depending on model type:

- **Linear Models** (Ridge, Lasso):  
  Applied **standard scaling** to numeric features using `StandardScaler`, improving convergence and accuracy.

- **Tree-Based Models** (Random Forest, XGBoost):  
  Trained **without scaling**, as these models are not sensitive to feature magnitude.

- **Categorical features** were one-hot encoded in all models using `OneHotEncoder`.

- The **Stacking Ensemble** architecture seamlessly combined both scaled and unscaled preprocessing pipelines.

This strategy ensured that each model operated under optimal preprocessing conditions.

---

##  Models and Evaluation

The following models were trained and evaluated using R², MAE, and RMSE:

| Model                 | R²     | MAE     | RMSE   |
|----------------------|--------|---------|--------|
| Ridge Regression      | 0.6256 | 78.70   | 126.93 |
| Lasso Regression      | 0.6231 | 78.64   | 127.35 |
| Random Forest (Tuned) | 0.6385 | 77.06   | 124.73 |
| **XGBoost (Tuned)**   | **0.6461** | **74.75** | **123.41** |
| Stacking Ensemble     | 0.6433 | 76.02   | 123.89 |

---

###  Key Insight

**XGBoost delivered the best performance**, explaining **64.6% of the variance** in payment amounts, followed closely by the Stacking Ensemble.  
This level of accuracy was achieved using just **three original features** — Insurance Provider, Service Type, and Miles — with the remaining predictive power unlocked through **feature engineering**.

This demonstrates that even with limited input data, a thoughtfully designed ML pipeline can yield reliable business insights.

---

##  Business Value

This model provides:

- Accurate payment estimates before submitting claims  
- Better financial forecasting for transport companies  
- Time-saving automation for cost estimation  
- Operational decision support for evaluating trip profitability

---

##  Deliverables

- `insurance_payment_model.py` —— final pipeline    
- `requirements.txt` — List of all dependencies   
- This `README.md` — Executive summary and usage guide  

---

##  Recommended Visuals (in `visuals/` folder)

- `feature_importance.png` — Top predictors visualized  
- `predicted_vs_actual.png` — Actual vs predicted payment scatter  
- `residuals.png` — Residual distribution plot  
- `mileage_vs_payment_boxplot.png` — Miles Category vs Payment Amount

> If you'd like, I can help you generate and save these from your notebook.

---

##  Conclusion

This project delivers a **business-ready machine learning solution** that can be integrated into real-world workflows like claim estimation, billing dashboards, or cost auditing systems. It balances performance, explainability, and data integrity using model-specific pipelines and proper cross-validation.

Despite limited input features, the project achieves **strong predictive performance** through thoughtful engineering and model tuning.

---

##  Author

**Amanuel Berhanu**  
*Data Analyst & Machine Learning Practitioner*  
[Upwork](https://www.upwork.com/freelancers/~01d81414fdc466b4e1) 

---
