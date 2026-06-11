from django.shortcuts import render

# Create your views here.
from .forms import PredictionForm
import numpy as np 
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt 
model=joblib.load("models/xgboost_model.pkl")
scaler=joblib.load("models/scaler.pkl")
explainer=shap.TreeExplainer(model)

def home(request):
    return render(
        request,"fraud_app/home.html"
    )
def predict(request):
    result=None
    fraud_prob=None
    risk_score=None
    risk_level=None
    top_features=[]
    if request.method=="POST":
        form=PredictionForm(request.POST)
        if form.is_valid():
            values=[
                form.cleaned_data[field]
                for field in form.cleaned_data
            ]
            arr=np.array(values).reshape(1,-1)
            scaled=scaler.transform(arr)
            prediction=model.predict(scaled)[0]
            prob=model.predict_proba(scaled)
            shap_values=explainer.shap_values(scaled)
            fraud_prob=round(
                prob[0][1]*100,2
                
            )
            risk_score=int(fraud_prob)
            if risk_score>=80:
                risk_level="Critical"
            elif risk_score>=60:
                risk_level="High"
            elif risk_score>=40:
                risk_level="Medium"
            else:
                risk_level="Low"
            result=(
                "Fraud"
                if prediction==1
                else "Legitimate"
                )
            feature_names=[ "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26","V27",
    "V28","Amount"]
            importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact": shap_values[0]
})
            importance_df = importance_df.reindex(
    importance_df["Impact"].abs().sort_values(
        ascending=False
    ).index
)

           

            top_features = importance_df.head(5)


    else:
        form=PredictionForm()
    return render(
            request,
            "fraud_app/predict.html",
            {
                "form":form,
                "result":result,
                "fraud_prob":fraud_prob ,
                "risk_score":risk_score,
                "risk_level":risk_level,
                "top_features": top_features.to_dict("records") if len(top_features) > 0 else []       
                }
    )