# IMDb Movie Reviews Sentiment Analysis 🎬📊

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.30.2-yellow)
![Gradio](https://img.shields.io/badge/Gradio-0.95.2-green)

A machine learning project that classifies IMDb movie reviews as **positive** or **negative** using traditional ML models and BERT, deployed via FastAPI.


## 🌟 Project Overview
This project performs **binary sentiment analysis** on 50,000 IMDb reviews using:

**1. Traditional ML**: TF-IDF + Logistic Regression (89% accuracy)

**2. Deep Learning**: Fine-tuned BERT (94% accuracy)

**3. API**: Gradio endpoint for real-time predictions

Dataset: [IMDb Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## ✨ Features
- **EDA**: Word clouds, n-gram analysis, class balance checks
- **Text Preprocessing**: HTML removal, lemmatization, stopword filtering
- **Models**:
  - `TfidfVectorizer` + Logistic Regression
  - `CountVectorizer` + Naive Bayes
  - Fine-tuned `bert-base-uncased`
- **API**: `/predict` endpoint with JSON I/O

