# Sentiment Based Product Recommendation

## Problem Statement
The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

Build a comprehensive machine learning project that analyzes product reviews and predicts sentiment (Positive, Neutral, or Negative) using various ML algorithms.

## Solution / Implementation
1. github link: https://github.com/shivkankan/Sentiment_Product_Recommendation_System
2. Heroku (Application is Live): https://sentimentproductrecommendation-b32b7dcaf731.herokuapp.com/
3. Google Drive link: https://drive.google.com/drive/folders/1skJhbLYymlgktQDMmzuUSh-laiF_-R37?usp=sharing

## Project Overview

This project implements a complete sentiment analysis pipeline including:
- Data cleaning and preprocessing
- Advanced text preprocessing (tokenization, lemmatization, spell correction)
- Multiple feature extraction techniques (BoW, TF-IDF)
- Training and comparison of 4 ML models
- Model evaluation and selection

The script will:
1. Load and clean the data
2. Preprocess the text
3. Extract features using TF-IDF
4. Train 4 ML models (Logistic Regression, Naive Bayes, Random Forest, XGBoost)
5. Evaluate and compare all models
6. Save the best model and generate visualizations
7. For a given Username Recommends 5 Products

## Deployment files
1. One 'model.py' file, which will contain only one ML model and only one recommendation system that is obtained from the previous steps along with the entire code to deploy the end-to-end project using Flask and Heroku
2. 'index.html' file, which includes the HTML code of the user interface
3. 'app.py' file, which is the Flask file to connect the backend ML model with the frontend HTML code
4. Supported pickle files, which have been generated while pickling the models
