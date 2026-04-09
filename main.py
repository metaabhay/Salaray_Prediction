import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")

st.title("Salary Prediction Dashboard")

# ---------------------------
# 1. LOAD DATA
# ---------------------------
st.header("1. Load Data")

if st.button("Load Data"):
    d = pd.read_csv("Salary_Data.csv")
    st.session_state["d"] = d

if "d" in st.session_state:
    d = st.session_state["d"]
    st.success("Data Loaded Successfully")
    st.dataframe(d.head())

    # ---------------------------
    # 2. FEATURE ENGINEERING
    # ---------------------------
    st.header("2. Feature Engineering & Cleaning")

    df = pd.read_csv("data.csv")

    # if "Gender" in df.columns:
    #     df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # df = df.dropna()

    # st.write("After Cleaning:")
    st.dataframe(df.head())

    st.info("""
### Data Cleaning & Feature Engineering

In this step, the dataset was prepared to ensure consistency, reliability, and suitability for model training.

1. **Handling Missing Values**
   - All rows with null or missing values were removed to prevent bias and inconsistencies in model learning.

2. **Categorical Encoding**
   - Categorical variables such as *Gender* were converted into numerical format using mapping (e.g., Male → 0, Female → 1) to make them usable for machine learning models.

3. **Text Standardization**
   - Columns like *Education Level* and *Job Title* were cleaned by:
     - Converting text to lowercase
     - Removing inconsistencies and variations
     - Grouping similar roles into broader categories (e.g., software roles, managerial roles, data-related roles)

4. **Outlier Treatment**
   - Numerical features such as *Age* were analyzed using the IQR (Interquartile Range) method.
   - Extreme outliers were removed to improve model stability and reduce noise.

5. **Data Consistency**
   - Ensured uniform data types across columns
   - Removed redundant or irrelevant entries

6. **Final Dataset**
   - The cleaned dataset is now structured, noise-free, and ready for:
     - Exploratory Data Analysis (EDA)
     - Feature selection
     - Model training

This preprocessing step significantly improves model performance and generalization.
""")

    # ---------------------------
    # 3. EDA GRAPHS
    # ---------------------------
    st.header("3. EDA - Relationship with Salary")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Reduce size globally
    plt.rcParams["figure.figsize"] = (6, 3)

    cols = df.columns

    for col in cols:
        if col != "Salary":

            st.subheader(f"{col} vs Salary")

            fig, ax = plt.subplots()

            if df[col].dtype == "object":
                # Categorical → avg salary
                avg_salary = df.groupby(col)["Salary"].mean().sort_values()

                sns.barplot(
                    x=avg_salary.index,
                    y=avg_salary.values,
                    ax=ax
                )

                ax.set_xlabel(col)
                ax.set_ylabel("Average Salary")

            else:
                # Numeric → bin then avg
                df["temp_bin"] = pd.cut(df[col], bins=5)

                avg_salary = df.groupby("temp_bin")["Salary"].mean()

                sns.barplot(
                    x=avg_salary.index.astype(str),
                    y=avg_salary.values,
                    ax=ax
                )

                ax.set_xlabel(col + " (binned)")
                ax.set_ylabel("Average Salary")
                ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

    # Clean temp column if created
    if "temp_bin" in df.columns:
        df.drop(columns=["temp_bin"], inplace=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
    st.pyplot(fig)

    # ---------------------------
    # 4. DATA SPLIT
    # ---------------------------
    st.header("4. Data Split")

    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    st.write("Train Shape:", X_train.shape)
    st.write("Test Shape:", X_test.shape)

    # ---------------------------
    # 5. MODEL SELECTION
    # ---------------------------
    st.header("5. Model Selection")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor( 
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        results[name] = r2

    st.write("Model R2 Scores:")
    st.write(results)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    st.success(f"Best Model: {best_model_name}")

    # ---------------------------
    # 6. K-FOLD VALIDATION
    # ---------------------------
    st.header("6. K-Fold Validation")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    kfold_scores = cross_val_score(best_model, X, y, cv=kf, scoring='r2')

    st.write("KFold Scores:", kfold_scores)
    st.write("Average Score:", kfold_scores.mean())

    # ---------------------------
    # 7. FINAL METRICS
    # ---------------------------
    st.header("7. Performance Metrics")

    import numpy as np
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        mean_absolute_error
    )

    # Train + Predict
    best_model.fit(X_train, y_train)
    final_preds = best_model.predict(X_test)

    # Core Metrics
    r2 = r2_score(y_test, final_preds)
    mse = mean_squared_error(y_test, final_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, final_preds)

    # Advanced Metrics
    mape = np.mean(np.abs((y_test - final_preds) / y_test)) * 100
    nrmse = rmse / y_test.mean()

    # -----------------------
    # DISPLAY (CLEAN UI)
    # -----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.2f}")


# ---------------------------
# 8. PREDICT SALARY
# ---------------------------
    import streamlit as st
    import joblib
    from n2 import transform_input

    # load model
    model = joblib.load("salary_model2.pkl")

    st.title("Salary Prediction App")

    # --- INPUTS ---
    age = st.number_input("Age", min_value=18, max_value=65, value=25)
    exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)

    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("education_encoded", ["High School", "Bachelor", "Master", "PhD"])
    job = st.selectbox("Job", ["IT", "Sales", "Others"])

    # --- PREDICT ---
    if st.button("Predict Salary"):
        user_input = {
            "Age": age,
            "Years of Experience": exp,
            "Gender": gender,
            "Education": education,
            "Job": job
        }

        processed = transform_input(user_input)
        prediction = model.predict(processed)

        st.success(f"Predicted Salary: {prediction[0]:,.2f}")