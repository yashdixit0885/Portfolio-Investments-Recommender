# News Analyzer Configuration

# Market hours (EST)
MARKET_HOURS = {
    'start': '09:30',
    'end': '16:00'
}

# News sources to monitor
NEWS_SOURCES = [
    'bloomberg',
    'reuters',
    'financial-times',
    'wall-street-journal'
]

# Keywords to monitor
KEYWORDS = [
    'stock market',
    'trading',
    'securities',
    'earnings',
    'merger',
    'acquisition',
    'IPO',
    'bankruptcy',
    'lawsuit',
    'investigation'
]

# Sentiment analysis thresholds
SENTIMENT_THRESHOLDS = {
    'strong_positive': 0.5,
    'strong_negative': -0.5,
    'moderate_positive': 0.3,
    'moderate_negative': -0.3
}

# Monitoring intervals (in minutes)
MONITORING_INTERVAL = 5

# Maximum number of articles to fetch per request
MAX_ARTICLES = 100

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}