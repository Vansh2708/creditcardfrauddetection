import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
def load_data(path):
    df=pd.read_csv(path)
    return df
def preprocess_data(df):
    X=df.drop("Class",axis=1)
    y=df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    smote=SMOTE(random_state=42)
    X_train,y_train=smote.fit_resample(X_train,y_train)
    return X_train,X_test,y_train,y_test,scaler