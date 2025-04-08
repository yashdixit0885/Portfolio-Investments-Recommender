# Research Analyzer Configuration

# Fundamental Analysis Thresholds
FUNDAMENTAL_THRESHOLDS = {
    'current_ratio_min': 1.5,
    'debt_to_equity_max': 1.0,
    'profit_margin_min': 0.15,
    'revenue_growth_min': 10.0,
    'earnings_growth_min': 10.0
}

# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'sma_short': 20,
    'sma_long': 50,
    'volume_ma_period': 20
}

# Scoring System
SCORING_WEIGHTS = {
    'fundamentals': 0.4,
    'technical': 0.3,
    'sentiment': 0.3
}

# Recommendation Thresholds
RECOMMENDATION_THRESHOLDS = {
    'high_confidence_min_score': 3,
    'moderate_confidence_min_score': 1,
    'low_confidence_max_score': 0
}

# Data Storage
DATA_FILES = {
    'opportunities': 'data/flagged_opportunities.json',
    'analysis': 'data/research_analysis.json',
    'recommendations': 'data/trade_recommendations.json'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Analysis Timeframes
TIMEFRAMES = {
    'short_term': '1mo',
    'medium_term': '3mo',
    'long_term': '1y'
}