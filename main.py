import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        font-size:16px;
        padding:10px;
    }
    .stMetric {
        background-color:#f5f5f5;
        padding:10px;
        border-radius:10px;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>💼 Salary Prediction Dashboard</h1>
<p style='text-align: center; font-size:18px;'>
Analyze data, train models, and predict salaries with an interactive ML pipeline 🚀
</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Navigation")
st.sidebar.info("Follow ML pipeline steps")

st.sidebar.markdown("""
1. 📊 EDA  
2. 🧹 Cleaning  
3. 🧠 Feature Selection  
4. 🤖 Training  
5. 📈 Performance  
6. 🧪 Predict  
""")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & EDA",
    "🧹 Cleaning",
    "🧠 Feature Selection",
    "🤖 Training",
    "📈 Performance",
    "🧪 Predict"
])

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")

# ================= TAB 1 =================
with tab1:
    st.header("Exploratory Data Analysis")

    target = st.selectbox("Select Target Variable", df.columns, index=len(df.columns)-1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Summary")
        st.dataframe(df.describe())

    with col2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Feature vs Target")

    feature = st.selectbox("Select Feature", [col for col in df.columns if col != target])

    fig, ax = plt.subplots()

    if df[feature].dtype == "object":
        avg = df.groupby(feature)[target].mean()
    else:
        bins = pd.cut(df[feature], bins=5)
        avg = df.groupby(bins)[target].mean()
        avg.index = [f"{int(i.left)}-{int(i.right)}" for i in avg.index]

    sns.barplot(x=avg.index, y=avg.values, ax=ax)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# ================= TAB 2 =================
with tab2:
    st.header("Data Cleaning & Engineering")

    st.info("Handle missing values, zeros, and outliers easily")

    action = st.radio(
        "Choose Action",
        ["Keep Data", "Remove Rows", "Impute Values"],
        horizontal=True
    )

    remove_outlier = st.checkbox("Apply Outlier Removal (IQR)")

    if remove_outlier:
        for col in df.select_dtypes(include=np.number).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    st.success(f"Updated Shape: {df.shape}")
    st.dataframe(df.head())

# ================= TAB 3 =================
with tab3:
    st.header("Feature Selection")

    method = st.radio("Method", ["All Features", "Top Correlated"])

    if method == "Top Correlated":
        corr = df.corr(numeric_only=True)["Salary"].abs().sort_values(ascending=False)
        selected = corr.index[1:6]
    else:
        selected = df.drop("Salary", axis=1).columns

    st.write("Selected Features:")
    st.code(list(selected))

    # Visualization
    fig, ax = plt.subplots()
    df.corr(numeric_only=True)["Salary"].sort_values().plot(kind='barh', ax=ax)
    st.pyplot(fig)

# ================= TAB 4 =================
with tab4:
    st.header("Model Training")

    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    test_size = st.slider("Test Size %", 10, 40, 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100
    )

    model_name = st.selectbox("Select Model", [
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

    if st.button("🚀 Start Training"):
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        st.session_state["model"] = model
        st.session_state["preds"] = preds
        st.session_state["y_test"] = y_test

        st.success("✅ Training Completed!")
        st.balloons()

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

        c1.metric("R² Score", f"{r2:.3f}")
        c2.metric("RMSE", f"{rmse:.0f}")
        c3.metric("MAE", f"{mae:.0f}")
        c4.metric("MSE", f"{mse:.0f}")

        # Cross Validation
        kf = KFold(n_splits=5, shuffle=True)
        score = cross_val_score(
            st.session_state["model"],
            df.drop("Salary", axis=1),
            df["Salary"],
            cv=kf,
            scoring="r2"
        ).mean()

        st.metric("Cross Validation Score", f"{score:.3f}")

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Train model first")

# ================= TAB 6 =================
with tab6:
    st.header("Predict Salary")

    st.subheader("👤 Personal Info")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 65, 25)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.subheader("💼 Professional Info")
    col3, col4 = st.columns(2)

    with col3:
        exp = st.slider("Experience", 0, 40, 2)

    with col4:
        job = st.selectbox("Job Role", ["IT", "Sales", "Others"])

    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])

    st.markdown("---")

    model = joblib.load("salary_model2.pkl")
    from n2 import transform_input

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

        st.markdown(f"""
        <div style='background-color:#e8f5e9;padding:20px;border-radius:10px'>
            <h2 style='color:#2e7d32;'>💰 Predicted Salary</h2>
            <h1>₹ {prediction[0]:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)
