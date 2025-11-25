from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
import os

def download_nltk_data():
    """Download NLTK data if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

# Download NLTK data
download_nltk_data()


class SentimentRecommenderModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        try:
            print("Loading sentiment classification model...")
            self.model = pickle.load(open(
                SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
            print("Model loaded successfully.")
            
            print("Loading TF-IDF vectorizer...")
            try:
                # First try with pickle.load
                self.vectorizer = pickle.load(open(
                    SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER, 'rb'))
            except Exception as pickle_error:
                print(f"Failed to load with pickle: {pickle_error}")
                try:
                    # Fallback to pd.read_pickle
                    print("Trying fallback method with pd.read_pickle...")
                    self.vectorizer = pd.read_pickle(
                        SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
                except Exception as pandas_error:
                    print(f"Failed to load with pd.read_pickle: {pandas_error}")
                    raise ValueError(f"Could not load vectorizer with either method. Pickle error: {pickle_error}, Pandas error: {pandas_error}")
            
            # Validate that the vectorizer is fitted
            if not hasattr(self.vectorizer, 'idf_'):
                raise ValueError("TF-IDF vectorizer is not fitted. Please check the vectorizer file.")
            
            print(f"Vectorizer loaded successfully with {len(self.vectorizer.vocabulary_)} features.")
            
            print("Loading user rating data...")
            self.user_final_rating = pickle.load(open(
                SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
            print("User rating data loaded successfully.")
            
            print("Loading sample data...")
            if not os.path.exists("data/sample30.csv"):
                raise FileNotFoundError("data/sample30.csv not found")
            self.data = pd.read_csv("data/sample30.csv")
            print("Sample data loaded successfully.")
            
            print("Loading cleaned data...")
            self.cleaned_data = pickle.load(open(
                SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))
            print("Cleaned data loaded successfully.")
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            print("SentimentRecommenderModel initialization complete!")
        except Exception as e:
            print(f"Error initializing SentimentRecommenderModel: {str(e)}")
            raise

    """function to get the top product 20 recommendations for the user"""

    def getRecommendationByUser(self, user):
        recommedations = []
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    """function to filter the product recommendations using the sentiment model and get the top 5 recommendations"""

    def getSentimentRecommendations(self, user):
        try:
            if (user in self.user_final_rating.index):
                # get the product recommedation using the trained ML model
                recommendations = list(
                    self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
                filtered_data = self.cleaned_data[self.cleaned_data.id.isin(
                    recommendations)]
                # preprocess the text before tranforming and predicting
                #filtered_data["reviews_text_cleaned"] = filtered_data["reviews_text"].apply(lambda x: self.preprocess_text(x))
                
                # Check if we have data to process
                if filtered_data.empty:
                    print(f"No data found for user {user} recommendations")
                    return None
                
                # Ensure we have the reviews_text_cleaned column
                if 'reviews_text_cleaned' not in filtered_data.columns:
                    print("Missing reviews_text_cleaned column in filtered_data")
                    return None
                
                # transfor the input data using saved tf-idf vectorizer
                print(f"Transforming {len(filtered_data)} reviews using TF-IDF vectorizer...")
                try:
                    X = self.vectorizer.transform(
                        filtered_data["reviews_text_cleaned"].values.astype(str))
                except Exception as e:
                    print(f"Error during TF-IDF transformation: {str(e)}")
                    return None
                
                filtered_data["predicted_sentiment"] = self.model.predict(X)
                temp = filtered_data[['id', 'predicted_sentiment']]
                temp_grouped = temp.groupby('id', as_index=False).count()
                temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(
                    temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
                temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
                temp_grouped['pos_sentiment_percent'] = np.round(
                    temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100, 2)
                sorted_products = temp_grouped.sort_values(
                    'pos_sentiment_percent', ascending=False)[0:5]
                return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])

            else:
                print(f"User name {user} doesn't exist")
                return None
        except Exception as e:
            print(f"Error in getSentimentRecommendations: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    """function to classify the sentiment to 1/0 - positive or negative - using the trained ML model"""

    def classify_sentiment(self, review_text):
        try:
            review_text = self.preprocess_text(review_text)
            X = self.vectorizer.transform([review_text])
            y_pred = self.model.predict(X)
            return y_pred
        except Exception as e:
            print(f"Error in classify_sentiment: {str(e)}")
            return None

    """function to preprocess the text before it's sent to ML model"""

    def preprocess_text(self, text):

        # cleaning the review text (lower, removing punctuation, numericals, whitespaces)
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        dictionary = "abc".maketrans('', '', string.punctuation)
        text = text.translate(dictionary)
        text = re.sub("\S*\d\S*", "", text)

        # remove stop-words and convert it to lemma
        text = self.lemma_text(text)
        return text

    """function to get the pos tag to derive the lemma form"""

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    """function to remove the stop words from the text"""

    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha()
                 and word not in self.stop_words]
        return " ".join(words)

    """function to derive the base lemma form of the text using the pos tag"""

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(
            self.remove_stopword(text)))  # Get position tags
        # Map the position tag and lemmatize the word/token
        words = [self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(
            tag[1])) for idx, tag in enumerate(word_pos_tags)]
        return " ".join(words)
