import joblib
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
from data_preprocessing import load_data,preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns
def evaluate():
    df=load_data("data/creditcard.csv")
    X_train,X_test,y_train,y_test,scaler=preprocess_data(df)
    model=joblib.load("models/fraud_model.pkl")
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    ##Fraud Vs Legitimate graph
    plt.figure()
    df["Class"].value_counts().plot(kind="bar")
    plt.title("Fraud vs Legitimate Transaction")
    plt.xlabel("Class (0=Legitimate,1=Fraud)")
    plt.ylabel("Count")
    plt.show()
    plt.savefig("models/fraud_vs_legitimate.png")
    #Confusion Matrix
    print("Confusion Matrix")
    print(confusion_matrix(y_test,y_pred))
    print("\nClassification Report")
    print(classification_report(y_test,y_pred))
    ##Matrix Graph
    cm=confusion_matrix(y_test,y_pred)
    plt.figure()
    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.show()
    plt.savefig("models/confusion_matrix.png")
    #Roc curve

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("models/roc_curve.png")
    #feature importance
    if hasattr(model, "feature_importances_"):

        features = df.drop("Class", axis=1).columns
        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        plt.figure()
        plt.bar(importance_df["Feature"][:10], importance_df["Importance"][:10])
        plt.xticks(rotation=90)
        plt.title("Top 10 Important Features")
        plt.show()
    
if __name__=="__main__":
    evaluate()