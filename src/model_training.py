import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from data_preprocessing import load_data, preprocess_data


def train_model():

    df = load_data("data/creditcard.csv")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=50, n_jobs=-1, class_weight="balanced", random_state=42
    )

    print("Training Random Forest Model...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = roc_auc_score(y_test, y_pred)

    print("ROC AUC Score:", score)

    # Save model
    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nModel saved successfully!")


if __name__ == "__main__":
    train_model()
