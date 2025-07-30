import nltk
nltk.download('punkt_tab')

import streamlit as st
import pandas as pd
import joblib
from preprocess import prepare_df


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

         # Display raw prediction probabilities
        st.subheader("Prediction Probabilities:")
        st.json({
            f"{cls}": f"{prob:.4f}"
            for cls, prob in zip(model.classes_, probabilities)
        })
