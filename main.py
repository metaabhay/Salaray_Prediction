import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide")

st.title("💼 Interactive ML Pipeline Dashboard")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
    "Data & EDA",
    "Cleaning & Engineering",
    "Feature Selection",
    "Model Training",
    "Performance",
    "Try Your Input"
])

# ---------------- LOAD ----------------
if "d" not in st.session_state:
    st.session_state["d"] = pd.read_csv("Salary_Data.csv")

d = st.session_state["d"]
df = pd.read_csv("data.csv")

# ================= TAB 1 =================
with tab1:
    st.header("Exploratory Data Analysis")

    target = st.selectbox("Select Target", df.columns, index=len(df.columns)-1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Summary")
        st.dataframe(df.describe())

    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Feature vs Target")

    target = "Salary"
    for col in df.columns:
        if col != target:
            fig, ax = plt.subplots()

            if df[col].dtype == "object":
                avg = df.groupby(col)[target].mean()

            else:
                bins = pd.cut(df[col], bins=5)
                avg = df.groupby(bins)[target].mean()

                # 👇 convert bins to readable ranges
                avg.index = [f"{int(i.left)} - {int(i.right)}" for i in avg.index]

            sns.barplot(x=avg.index, y=avg.values, ax=ax)

            ax.set_xlabel(col)
            ax.set_ylabel(target)
            ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

# ================= TAB 2 =================
with tab2:
    st.header("Data Cleaning & Engineering")

    cols = st.multiselect(
        "Columns to check zeros",
        df.columns,
        default=list(df.columns[:5])
    )

    action = st.radio("Action", ["Keep", "Remove", "Impute"])

    st.subheader("Outlier Removal")
    remove_outlier = st.checkbox("Apply IQR")

    if remove_outlier:
        for col in df.select_dtypes(include=np.number).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    st.success(f"Shape: {df.shape}")
    st.dataframe(df.head())

# ================= TAB 3 =================
with tab3:
    st.header("Feature Selection")

    method = st.radio("Method", [
        "All Features",
        "Top Correlated",
    ])

    if method == "Top Correlated":
        corr = df.corr(numeric_only=True)["Salary"].abs().sort_values(ascending=False)
        selected = corr.index[1:6]
    else:
        selected = df.drop("Salary", axis=1).columns

    st.write("Selected Features:")
    st.code(list(selected))

    st.dataframe(df[selected].head())

# ================= TAB 4 =================
with tab4:
    st.header("Model Training")

    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    test_size = st.slider("Test Size %", 10, 40, 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100
    )

    model_name = st.selectbox("Model", [
        "Linear Regression",
        "Random Forest",
        "Decision Tree"
    ])

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
    else:
        model = DecisionTreeRegressor()

    if st.button("Start Training"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.session_state["model"] = model
        st.session_state["preds"] = preds
        st.session_state["y_test"] = y_test

        st.success("Training Done")

# ================= TAB 5 =================
with tab5:
    st.header("Performance")

    if "model" in st.session_state:

        preds = st.session_state["preds"]
        y_test = st.session_state["y_test"]

        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("R²", f"{r2:.3f}")
        c2.metric("RMSE", f"{rmse:.0f}")
        c3.metric("MAE", f"{mae:.0f}")
        c4.metric("MSE", f"{mse:.0f}")

        # KFold
        kf = KFold(n_splits=5, shuffle=True)
        score = cross_val_score(
            st.session_state["model"],
            df.drop("Salary", axis=1),
            df["Salary"],
            cv=kf,
            scoring="r2"
        ).mean()

        st.metric("Cross Val Score", f"{score:.3f}")

        # prediction vs actual plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    else:
        st.warning("Train model first")

# ---------------------------
# 8. USER INPUT (PIPELINE STYLE)
# ---------------------------
with tab6:
    st.header("8. Try Your Own Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 65, 25)

    with col2:
        exp = st.slider("Years of Experience", 0, 40, 2)

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])

    col4, col5 = st.columns(2)

    with col4:
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])

    with col5:
        job = st.selectbox("Job Role", ["IT", "Sales", "Others"])

    st.markdown("---")

    # load model once
    import joblib
    from n2 import transform_input

    model = joblib.load("salary_model2.pkl")

    if st.button("🚀 Predict Salary"):
        user_input = {
            "Age": age,
            "Years of Experience": exp,
            "Gender": gender,
            "Education": education,
            "Job": job
        }

        processed = transform_input(user_input)
        prediction = model.predict(processed)

        st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.0f}")
