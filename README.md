#  Movie Review Sentiment Analysis App

This project is a **Streamlit-based web application** that predicts the **sentiment** (Positive or Negative) of a movie review using a Logistic Regression model trained on IMDB-style text data.

---

##  Features

- Input any movie review and get instant sentiment feedback.
- Text preprocessing: tokenization, stopword removal, lemmatization.
- Additional features: word count and review length.
- Logistic Regression pipeline with `TfidfVectorizer` + `StandardScaler` with `ColumnTransformer`.
- Streamlit UI for easy interaction.

---

##  Exploratory Data Analysis (EDA)

several insights were drawn from the dataset:

- **Word Count Distribution by Sentiment**  
  Analyzed how the length of reviews varies between positive and negative classes.

- **Class Imbalance**  
  Checked the distribution of sentiment labels to ensure balanced learning.

---

## 📈 Model Evaluation & Interpretation

- **Model Used:** Logistic Regression
- **Vectorization:** `TfidfVectorizer` (Top 5000 features)
- **Numerical Features:** Word count and review length scaled using `StandardScaler`.
- **Feature Importance:**
  - Extracted and visualized top features influencing predictions.
  - Identified words with strongest positive and negative coefficients.

Example:
Positive indicators: 'amazing', 'best', 'love', 'great'
Negative indicators: 'worst', 'boring', 'waste', 'bad'


- **Performance Metrics:** Accuracy, precision, recall, and F1-score were computed

---

## 🧠 How It Works

1. **Preprocessing** (`preprocess.py`):
   - Cleans the input review.
   - Generates `cleaned_text`, `word_count`, and `review_length`.

2. **Model Building + EDA + Evaluation** (`imdb_sentiment_analysis.ipynb`):
   - Text vectorized with TF-IDF.
   - Numerical features scaled.
   - Logistic Regression for sentiment classification.

3. **App Interface** (`app.py`):
   - User inputs a review.
   - Review is preprocessed and fed into the model.
   - Sentiment label and review stats are shown.

---

##  File Structure

├── app.py # Streamlit app
├── preprocess.py # Preprocessing logic
├── imdb_sentiment_analysis.ipynb # Model Building + EDA & insights
├── README.md # Project documentation
