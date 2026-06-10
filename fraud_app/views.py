from django.shortcuts import render

# Create your views here.
from .forms import PredictionForm
import numpy as np 
import joblib
model=joblib.load("models/xgboost_model.pkl")
scaler=joblib.load("models/scaler.pkl")
def home(request):
    return render(
        request,"fraud_app/home.html"
    )
def predict(request):
    result=None
    fraud_prob=None
    risk_score=None
    risk_level=None
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
                "risk_level":risk_level         
                }
    )