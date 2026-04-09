import pandas as pd
import joblib

pt = joblib.load("pt.pkl")

def transform_input(data):
    df = pd.DataFrame([data])

    # --- Gender ---
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # --- Education (binary → education_encoded) ---
    df["education_encoded"] = df["Education"].str.lower().apply(
        lambda x: 0 if x in ["high school", "bachelor"] else 1
    )

    # --- Job (one-hot EXACT names) ---
    df["IT"] = (df["Job"] == "IT").astype(int)
    df["Sales"] = (df["Job"] == "SALES").astype(int)
    df["Others"] = (df["Job"] == "OTHERS").astype(int)

    # --- Yeo-Johnson ---
    df[["Age", "Years of Experience"]] = pt.transform(
        df[["Age", "Years of Experience"]]
    )

    # --- FINAL ORDER (CRITICAL) ---
    df = df[
        ['Age', 'Gender', 'Years of Experience', 'education_encoded', 'IT', 'Sales', 'Others']
    ]

    return df