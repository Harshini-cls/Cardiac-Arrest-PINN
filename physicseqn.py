import pandas as pd
import numpy as np

df = pd.read_excel("Cardiac_Dataset.xlsx")

df["Shock_Index"] = df["Heart Rate"] / df["Systolic Blood Pressure"]
df["MAP"] = (df["Systolic Blood Pressure"] + 2 * df["DBP"]) / 3
df["Oxygen_Deficit"] = 100 - df["SpO₂"]

df["Ion_Imbalance_Score"] = (
    abs(df["Potassium (K⁺)"] - 4.2)
    + abs(df["Sodium (Na⁺)"] - 140) / 10
    + abs(df["Calcium (Ca²⁺)"] - 9.5)
)

df["Acid_Base_Stress"] = abs(df["pH"] - 7.4) + df["Lactate"] / 10
df["Renal_Stress"] = df["Creatinine"] / 2

df["Perfusion_Risk"] = (
    df["Shock_Index"]
    + df["Oxygen_Deficit"] / 100
    + df["Acid_Base_Stress"]
    + df["Ion_Imbalance_Score"]
)

df["Physics_Risk_Score"] = (
    0.25 * df["Shock_Index"]
    + 0.20 * df["Oxygen_Deficit"] / 100
    + 0.25 * df["Ion_Imbalance_Score"]
    + 0.20 * df["Acid_Base_Stress"]
    + 0.10 * df["Renal_Stress"]
)

df["Risk_Label"] = df["Cardiac Arrest Risk label"].map({
    "Low Risk": 0,
    "Moderate Risk": 1,
    "High Risk": 1
})

df.to_csv("cardiac_physics_features.csv", index=False)

print("Physics features created")
print("Saved: cardiac_physics_features.csv")
print(df["Risk_Label"].value_counts())