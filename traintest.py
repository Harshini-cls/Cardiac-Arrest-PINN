import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("cardiac_physics_features.csv")

drop_cols = [
    "Patient_ID",
    "Region_Label",
    "Cardiac Arrest Risk label",
    "Risk_Label"
]

X = df.drop(columns=drop_cols)
X = X.select_dtypes(include=["int64", "float64"])
y = df["Risk_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

joblib.dump(scaler, "scaler.pkl")

print("Train-test split completed")
print("X_train.csv saved")
print("X_test.csv saved")
print("y_train.csv saved")
print("y_test.csv saved")