import joblib 
import pandas as pd
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)
from data_preprocessing import load_data,preprocess_data
def compare_models():
    print("Loading Dataset...")
    df=load_data("data/creditcard.csv")
    X_train,X_test,y_train,y_test,scaler=preprocess_data(df)
    rf_model=joblib.load("models/fraud_model.pkl")
    xgb_model=joblib.load("models/xgboost_model.pkl")
    results=[]
    models={
        "Random Forest":rf_model,
        "XGBoost":xgb_model
    }
    for name,model in models.items():
        y_pred=model.predict(X_test)
        y_prob=model.predict_proba(X_test)[:,1]
        results.append({
            "Model":name,
            "Accuracy":accuracy_score(y_test,y_pred),
            "Precision":precision_score(y_test,y_pred),
            "Recall":recall_score(y_test,y_pred),
            "F1 Score":f1_score(y_test,y_pred),
            "ROC-AUC":roc_auc_score(y_test,y_prob)
            
            
        })
    results_df=pd.DataFrame(results)
    print("\nModel Comaprison\n")
    print(results_df)
    results_df.to_csv(
        "models/model_comparison.csv",index=False
    )
    print(
        "\nSaved to models/model_comparison.csv"
    )
if __name__=="__main__":
    compare_models()