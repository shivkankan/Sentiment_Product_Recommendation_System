from flask import Flask, request, render_template
import os
import sys
from model import SentimentRecommenderModel


app = Flask(__name__)

# Initialize the model with error handling
try:
    sentiment_model = SentimentRecommenderModel()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Traceback:", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    try:
        user = request.form['userName']
        user = user.lower()
        items = sentiment_model.getSentimentRecommendations(user)

        if(items is not None):
            print(f"retrieving items....{len(items)}")
            print(items)
            return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
        else:
            return render_template("index.html", message="User Name doesn't exists, No product recommendations at this point of time!")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return render_template("index.html", message="An error occurred while processing your request. Please try again.")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
