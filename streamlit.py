import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Title
st.title("📊 Student Performance Prediction Dashboard")
st.write("Predict the **Performance Index** of a student using study and lifestyle factors.")

# Load dataset
data = pd.read_csv("Student_Performance.csv")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Features
X = data[['Hours Studied','Previous Scores','Sleep Hours','Sample Question Papers Practiced']]
y = data['Performance Index']

# Train model
model = LinearRegression()
model.fit(X,y)

# Sidebar inputs
st.sidebar.header("🎯 Enter Student Details")

hours = st.sidebar.slider("Hours Studied",0,10,5)
previous = st.sidebar.slider("Previous Scores",0,100,60)
sleep = st.sidebar.slider("Sleep Hours",0,10,7)
papers = st.sidebar.slider("Sample Papers Practiced",0,10,3)

# Prediction
if st.sidebar.button("Predict Performance"):

    prediction = model.predict([[hours,previous,sleep,papers]])

    # Limit to realistic values
    prediction_value = round(np.clip(prediction[0],0,100))

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Performance Index",
        value=prediction_value
    )

# Visualization Section
st.subheader("📈 Relationship Between Variables")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=data['Hours Studied'], y=data['Performance Index'], color="orange")
    sns.regplot(x=data['Hours Studied'], y=data['Performance Index'], scatter=False, color="blue")
    plt.title("Hours Studied vs Performance")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=data['Previous Scores'], y=data['Performance Index'], color="green")
    sns.regplot(x=data['Previous Scores'], y=data['Performance Index'], scatter=False, color="blue")
    plt.title("Previous Scores vs Performance")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=data['Sleep Hours'], y=data['Performance Index'], color="purple")
    sns.regplot(x=data['Sleep Hours'], y=data['Performance Index'], scatter=False, color="blue")
    plt.title("Sleep Hours vs Performance")
    st.pyplot(fig3)

with col4:
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=data['Sample Question Papers Practiced'], y=data['Performance Index'], color="red")
    sns.regplot(x=data['Sample Question Papers Practiced'], y=data['Performance Index'], scatter=False, color="blue")
    plt.title("Practice Papers vs Performance")
    st.pyplot(fig4)