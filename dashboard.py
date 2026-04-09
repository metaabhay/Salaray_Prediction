import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("Salary Prediction Dashboard")

df = pd.read_csv("Salary_Data.csv")

st.subheader("Raw Data")
st.dataframe(df)

st.subheader("EDA")
st.write(df.describe())

fig, ax = plt.subplots()
df['Salary'].hist(ax=ax)
st.pyplot(fig)

st.subheader("Data Cleaning")
df = df.dropna()

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

st.write(df.head())

st.subheader("Feature Selection")
X = df.drop("Salary", axis=1)
y = df["Salary"]

st.write("Features:", X.columns.tolist())

st.subheader("Train Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

st.write("Train size:", X_train.shape)
st.write("Test size:", X_test.shape)

st.subheader("Model Training")
model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("K-Fold Validation")
kf = KFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=kf)
st.write("K-Fold Scores:", scores)
st.write("Average Score:", np.mean(scores))

st.subheader("Performance Metrics")
y_pred = model.predict(X_test)

st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R2 Score:", r2_score(y_test, y_pred))

st.subheader("Prediction")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write("Predicted Salary:", prediction[0])