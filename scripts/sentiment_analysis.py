import logging
import os
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    logger.info("NLTK resources downloaded successfully")

class SentimentAnalyzer:
    """Analyzes sentiment of news and social media for crypto assets"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            self.initialized = False
    
    def analyze_text(self, text):
        """Analyze sentiment of text"""
        if not self.initialized:
            logger.error("Sentiment analyzer not initialized")
            return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
        
        try:
            return self.analyzer.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
    
    def analyze_news(self, news_items):
        """Analyze sentiment of news items"""
        if not news_items:
            return {"compound": 0, "positive": 0, "negative": 0, "neutral": 0}
        
        compound_scores = []
        for item in news_items:
            text = item.get("title", "") + " " + item.get("description", "")
            scores = self.analyze_text(text)
            compound_scores.append(scores["compound"])
        
        # Average compound score
        avg_compound = sum(compound_scores) / len(compound_scores)
        
        # Determine sentiment
        if avg_compound >= 0.05:
            sentiment = "positive"
        elif avg_compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "compound": avg_compound,
            "sentiment": sentiment,
            "sample_size": len(news_items)
        }

# For testing
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test with sample text
    sample_text = "Ethereum price surges to new all-time high as DeFi adoption grows"
    result = analyzer.analyze_text(sample_text)
    print(f"Sentiment analysis for: '{sample_text}'")
    print(f"Result: {result}")
