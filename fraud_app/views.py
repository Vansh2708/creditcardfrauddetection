from django.shortcuts import render


from .forms import PredictionForm
import numpy as np 
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt 
from .models import PredictionHistory
from collections import Counter
from django.http import HttpResponse
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import csv

model=joblib.load("models/xgboost_model.pkl")
scaler=joblib.load("models/scaler.pkl")
explainer=shap.TreeExplainer(model)

#Home
def home(request):
    return render(
        request,"fraud_app/home.html"
    )
#Prediction
def predict(request):
    result=None
    fraud_prob=None
    risk_score=None
    risk_level=None
    top_features=[]
    if request.method=="POST":
        form=PredictionForm(request.POST)
        if form.is_valid():
            values = [
    form.cleaned_data["Time"],

    form.cleaned_data["V1"],
    form.cleaned_data["V2"],
    form.cleaned_data["V3"],
    form.cleaned_data["V4"],
    form.cleaned_data["V5"],
    form.cleaned_data["V6"],
    form.cleaned_data["V7"],
    form.cleaned_data["V8"],
    form.cleaned_data["V9"],
    form.cleaned_data["V10"],
    form.cleaned_data["V11"],
    form.cleaned_data["V12"],
    form.cleaned_data["V13"],
    form.cleaned_data["V14"],
    form.cleaned_data["V15"],
    form.cleaned_data["V16"],
    form.cleaned_data["V17"],
    form.cleaned_data["V18"],
    form.cleaned_data["V19"],
    form.cleaned_data["V20"],
    form.cleaned_data["V21"],
    form.cleaned_data["V22"],
    form.cleaned_data["V23"],
    form.cleaned_data["V24"],
    form.cleaned_data["V25"],
    form.cleaned_data["V26"],
    form.cleaned_data["V27"],
    form.cleaned_data["V28"],

    form.cleaned_data["Amount"]
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
            PredictionHistory.objects.create(
                result=result,
                fraud_probability=fraud_prob,
                risk_score=risk_score,
                risk_level=risk_level
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
    
#Dashboard
def dashboard(request):
    predictions=PredictionHistory.objects.all().order_by("-created_at")
    total=predictions.count()
    fraud_count=predictions.filter(
        result="Fraud"
    ).count()
    legit_count=predictions.filter(
        result="Legitimate"
    ).count()
    avg_risk=0
    if total>0:
        avg_risk=sum(
            p.risk_score for p in predictions
        )/total
    #Pie Chart
    plt.figure(figsize=(5,5))

    plt.pie(
        [fraud_count, legit_count],
        labels=["Fraud", "Legitimate"],
        autopct="%1.1f%%"
    )

    plt.title("Fraud vs Legitimate Transactions")

    plt.savefig("static/images/fraud_pie.png")

    plt.close()
    
    #Risk Level Chart
    risk_levels = predictions.values_list(
        "risk_level",
        flat=True
    )

    counts = Counter(risk_levels)

    plt.figure(figsize=(6,4))

    plt.bar(
        counts.keys(),
        counts.values()
    )

    plt.title("Risk Level Distribution")
    plt.xlabel("Risk Level")

    plt.ylabel("Count")

    plt.savefig("static/images/risk_chart.png")

    plt.close()


    context={
        "total": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "avg_risk": round(avg_risk,2),
        "predictions": predictions[:10]
    }
    return render(
        request,
     "fraud_app/dashboard.html",
     context
    )
#Report Generation
def download_report(request):

    response = HttpResponse(
        content_type='application/pdf'
    )

    response['Content-Disposition'] = (
        'attachment; filename="fraud_report.pdf"'
    )

    doc = SimpleDocTemplate(response)

    styles = getSampleStyleSheet()

    content = []

    predictions = PredictionHistory.objects.all()

    total = predictions.count()

    fraud_count = predictions.filter(
        result="Fraud"
    ).count()

    legit_count = predictions.filter(
        result="Legitimate"
    ).count()

    avg_risk = 0

    if total > 0:
        avg_risk = round(
            sum(p.risk_score for p in predictions) / total,
            2
        )

    # Title

    content.append(
        Paragraph(
            "Credit Card Fraud Detection System",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 10))

    content.append(
        Paragraph(
            "AI Powered Transaction Risk Analysis Report",
            styles["Heading2"]
        )
    )

    content.append(Spacer(1, 20))

    # Summary

    content.append(
        Paragraph(
            f"<b>Total Predictions:</b> {total}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Fraud Transactions:</b> {fraud_count}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Legitimate Transactions:</b> {legit_count}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Average Risk Score:</b> {avg_risk}",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    # Recent Predictions Table

    content.append(
        Paragraph(
            "Recent Prediction Records",
            styles["Heading2"]
        )
    )

    content.append(Spacer(1, 10))

    data = [
        [
            "Result",
            "Risk Score",
            "Risk Level",
            "Date"
        ]
    ]

    for p in predictions[:10]:

        data.append(
            [
                p.result,
                str(p.risk_score),
                p.risk_level,
                p.created_at.strftime(
                    "%d-%m-%Y %H:%M"
                )
            ]
        )

    table = Table(data)

    table.setStyle(
        TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),1,colors.black),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("BACKGROUND",(0,1),(-1,-1),colors.whitesmoke),
        ])
    )

    content.append(table)

    content.append(Spacer(1,20))

    content.append(
        Paragraph(
            "Generated by Credit Card Fraud Detection System",
            styles["Italic"]
        )
    )

    doc.build(content)

    return response
#CSV Report
def export_csv(request):

    response = HttpResponse(
        content_type="text/csv"
    )

    response[
        "Content-Disposition"
    ] = 'attachment; filename="fraud_prediction_history.csv"'

    writer = csv.writer(response)

    writer.writerow([
        "Credit Card Fraud Detection System"
    ])

    writer.writerow([])

    writer.writerow([
        "Result",
        "Risk Score",
        "Risk Level",
        "Fraud Probability",
        "Date"
    ])

    predictions = PredictionHistory.objects.all()

    for p in predictions:

        writer.writerow([
            p.result,
            p.risk_score,
            p.risk_level,
            p.fraud_probability,
            p.created_at.strftime(
                "%d-%m-%Y %H:%M"
            )
        ])

    return response