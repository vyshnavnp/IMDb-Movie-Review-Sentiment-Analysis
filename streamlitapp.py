import nltk
nltk.download('punkt_tab')

import streamlit as st
import pandas as pd
import joblib
from preprocess import prepare_df

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,confusion_matrix
from sklearn.pipeline import make_pipeline

model = joblib.load("logisticregression.pkl")

st.title("Movie Review Sentiment Analysis")

st.write("Enter your review for the movie")
user_input = st.text_area("How do you feel about this movie?",height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please Enter a Review")
    else:
        df = pd.DataFrame({"review":[user_input]})
        processed_df = prepare_df(df)

        prediction = model.predict(processed_df)[0]
        label = "Positive 😀" if prediction == "positive" else "Negative 😞"

        st.subheader("Prediction:")
        st.success(f"This Review is *{label}*")
