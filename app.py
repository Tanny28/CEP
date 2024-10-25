import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Disease Prediction Dashboard", layout="wide", page_icon="🩺")

# Canva Embed Code (replace with your actual Canva embed code)
st.markdown(
    """
    <div style="text-align: center;">
        <iframe src="https://www.canva.com/design/DAGUIKuxKWQ/4ieiuxWMEWG164U4N80sNw/view" width="800" height="400" frameborder="0" allowfullscreen></iframe>
    </div>
    """,
    unsafe_allow_html=True
)

# Disease Prediction Section
st.title("Disease Prediction Dashboard")
st.write("""
This interactive dashboard uses machine learning to predict the likelihood of various diseases based on selected datasets.
""")

# File Upload Section
uploaded_file = st.file_uploader("Upload a disease dataset (CSV format)", type=["csv"])
if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Overview")
    st.write("**Dataset Shape:**", df.shape)
    st.write("**First 5 rows:**")
    st.dataframe(df.head())
    
    # Disease Prediction Model (e.g., RandomForest for demonstration)
    st.subheader("Model Training & Prediction")
    target_variable = st.selectbox("Select Target Variable for Prediction", df.columns)
    
    if target_variable:
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Model Evaluation
        st.write("**Model Evaluation**")
        st.text(classification_report(y_test, model.predict(X_test)))

        # Feature Importance
        st.write("**Feature Importance**")
        feature_importance = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance, y=X.columns, ax=ax)
        st.pyplot(fig)

# Footer
st.write("## About")
st.write("""
This app is designed for analyzing health datasets and predicting disease outcomes using machine learning models.
""")
