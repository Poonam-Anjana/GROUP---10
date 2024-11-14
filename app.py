from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    sentiment_scores = sia.polarity_scores(text)
    polarity = sentiment_scores['compound']

    if polarity > 0.05:
        result = 'Positive Sentiment 😊'
    elif polarity < -0.05:
        result = 'Negative Sentiment 😞'
    else:
        result = 'Neutral Sentiment 😐'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
