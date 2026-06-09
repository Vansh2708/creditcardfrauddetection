import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from data_preprocessing import load_data, preprocess_data


def train_xgboost():

    print("Loading Dataset...")

    df = load_data("data/creditcard.csv")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("Training XGBoost Model...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    roc_score = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_score:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/xgboost_model.pkl")

    print("\nXGBoost model saved successfully!")


if __name__ == "__main__":
    train_xgboost()