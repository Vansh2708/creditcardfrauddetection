import joblib
import shap
import pandas as pd
model=joblib.load("models/xgboost_model.pkl")
explainer=shap.TreeExplainer(model)
def get_shap_values(data):
    shape_values=explainer.shap_values(data)
    return shape_values