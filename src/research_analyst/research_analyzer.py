import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
import logging
import re
from utils.common import setup_logging, get_current_time, save_to_json, load_from_json
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class ResearchAnalyzer:
    def __init__(self):
        """Initialize the ResearchAnalyzer."""
        self.logger = setup_logging('research_analyzer')
        self.opportunities_file = 'data/flagged_opportunities.json'
        self.analysis_file = 'data/research_analysis.json'
        self.trade_recommendations_file = 'data/trade_recommendations.json'
        
        # Risk tolerance parameters
        self.risk_free_rate = 0.04  # 4% risk-free rate
        self.min_sharpe_ratio = 1.0  # Minimum acceptable Sharpe ratio
        self.max_beta = 2.0  # Maximum acceptable beta
        self.min_alpha = 0.02  # Minimum acceptable alpha (2%)
        
        # Load company keywords for better ticker extraction
        self.company_keywords = self._load_company_keywords()
        
        # Initialize sentiment keywords
        self.positive_keywords = {
            'bullish', 'growth', 'positive', 'up', 'gain', 'profit', 'success', 
            'strong', 'improve', 'increase', 'rise', 'higher', 'better', 'excellent',
            'outperform', 'buy', 'opportunity', 'potential', 'promising', 'solid',
            'robust', 'thrive', 'boom', 'surge', 'soar', 'jump', 'leap', 'boost',
            'accelerate', 'expand', 'develop', 'innovate', 'breakthrough', 'leader',
            'premium', 'quality', 'efficient', 'effective', 'optimize', 'maximize',
            'outstanding', 'superior', 'excellent', 'premium', 'premier', 'top',
            'best', 'leading', 'pioneer', 'revolutionary', 'transformative'
        }
        
        self.negative_keywords = {
            'bearish', 'decline', 'negative', 'down', 'loss', 'fail', 'weak',
            'worse', 'decrease', 'fall', 'lower', 'poor', 'bad', 'terrible',
            'underperform', 'sell', 'risk', 'threat', 'concern', 'uncertain',
            'volatile', 'unstable', 'trouble', 'problem', 'issue', 'challenge',
            'difficulty', 'struggle', 'suffer', 'hurt', 'damage', 'harm',
            'decline', 'deteriorate', 'worsen', 'weaken', 'undermine', 'compromise',
            'inferior', 'subpar', 'mediocre', 'average', 'ordinary', 'common',
            'lag', 'behind', 'trailing', 'falling', 'dropping', 'sinking'
        }
        
    def _load_company_keywords(self) -> Set[str]:
        """Load company names and tickers for better extraction"""
        try:
            # Try to load from a file if it exists
            if os.path.exists('data/company_keywords.json'):
                return set(load_from_json('data/company_keywords.json'))
            
            # Otherwise, create a basic set of common stock symbols
            return {
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
                'JNJ', 'PG', 'MA', 'HD', 'BAC', 'XOM', 'DIS', 'ADBE', 'NFLX', 'PYPL',
                'INTC', 'VZ', 'CSCO', 'PFE', 'KO', 'PEP', 'ABT', 'TMO', 'AVGO', 'QCOM',
                'TXN', 'INTU', 'AMAT', 'AMD', 'MU', 'CRM', 'ORCL', 'IBM', 'ACN', 'NOW'
            }
        except Exception as e:
            self.logger.error(f"Error loading company keywords: {str(e)}")
            return set()

    def extract_ticker_from_news(self, news_item: Dict[str, Any]) -> Optional[str]:
        """Extract potential ticker symbols from news content"""
        # Get the text to analyze
        title = news_item.get('title', '')
        description = news_item.get('description', '')
        text = f"{title} {description}"
        
        # Common stock symbols we want to identify
        major_stocks = {
            'AAPL': ['Apple', 'iPhone maker'],
            'MSFT': ['Microsoft'],
            'GOOGL': ['Google', 'Alphabet'],
            'AMZN': ['Amazon'],
            'META': ['Meta', 'Facebook'],
            'TSLA': ['Tesla'],
            'NVDA': ['Nvidia'],
            'JPM': ['JPMorgan', 'JP Morgan'],
            'V': ['Visa'],
            'WMT': ['Walmart'],
            'DIS': ['Disney'],
            'NFLX': ['Netflix'],
            'INTC': ['Intel'],
            'AMD': ['AMD', 'Advanced Micro Devices'],
            'CRM': ['Salesforce'],
            'ORCL': ['Oracle'],
            'IBM': ['IBM', 'International Business Machines'],
            'CSCO': ['Cisco'],
            'ADBE': ['Adobe'],
            'QCOM': ['Qualcomm']
        }
        
        # Look for exact ticker matches with $ prefix
        ticker_pattern = r'\$([A-Z]{1,5}(?:\.[A-Z]{1,2})?)'
        matches = re.findall(ticker_pattern, text)
        if matches and matches[0] in self.company_keywords:
            return matches[0]
        
        # Look for company names followed by tickers in parentheses
        company_ticker_pattern = r'([A-Za-z\s]+)\s*\(([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)'
        matches = re.findall(company_ticker_pattern, text)
        if matches and matches[0][1] in self.company_keywords:
            return matches[0][1]
        
        # Look for major company names and return their tickers
        for ticker, names in major_stocks.items():
            for name in names:
                if name in text:
                    return ticker
        
        # Look for standalone tickers that are in our company keywords
        words = text.split()
        potential_tickers = [word for word in words 
                           if word.isupper() and len(word) <= 5 
                           and word in self.company_keywords
                           and not word in ['US', 'USA', 'UK', 'EU', 'CEO', 'IPO', 'AI', 'GDP', 'TPU', 'GPU', 'CPU', 'RAM', 'ROM']]
        
        if potential_tickers:
            # Validate the ticker before returning it
            if self._validate_ticker(potential_tickers[0]):
                return potential_tickers[0]
        
        # Check source-specific patterns
        source = news_item.get('source', '')
        url = news_item.get('url', '')
        
        if source == 'Seeking Alpha':
            # Seeking Alpha often has tickers in the URL
            url_ticker = re.search(r'/([A-Z]{1,5}(?:\.[A-Z]{1,2})?)-', url)
            if url_ticker and url_ticker.group(1) in self.company_keywords:
                return url_ticker.group(1)
        
        return None

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch company information using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'short_ratio': info.get('shortRatio'),
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets')
            }
        except Exception as e:
            self.logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            return {}

    def analyze_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Analyze company fundamentals"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            # Calculate key metrics
            current_ratio = balance_sheet.loc['Total Current Assets'].iloc[0] / balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index else 0
            debt_to_equity = balance_sheet.loc['Total Debt'].iloc[0] / balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index else 0
            profit_margin = income_stmt.loc['Net Income'].iloc[0] / income_stmt.loc['Total Revenue'].iloc[0] if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index else 0
            
            return {
                'current_ratio': current_ratio,
                'debt_to_equity': debt_to_equity,
                'profit_margin': profit_margin,
                'revenue_growth': self._calculate_growth_rate(income_stmt.loc['Total Revenue']) if 'Total Revenue' in income_stmt.index else 0,
                'earnings_growth': self._calculate_growth_rate(income_stmt.loc['Net Income']) if 'Net Income' in income_stmt.index else 0,
                'free_cash_flow': cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0,
                'operating_cash_flow': cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else 0
            }
        except Exception as e:
            self.logger.error(f"Error analyzing fundamentals for {ticker}: {str(e)}")
            return {}

    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate year-over-year growth rate"""
        if len(series) < 2:
            return 0
        return ((series.iloc[0] - series.iloc[1]) / series.iloc[1]) * 100

    def analyze_market_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze market sentiment indicators"""
        try:
            stock = yf.Ticker(ticker)
            # Get more historical data for better analysis
            hist = stock.history(period='6mo')
            
            if hist.empty:
                return {}
                
            # Calculate technical indicators
            current_price = hist['Close'].iloc[-1]
            
            # Moving averages
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Exponential moving averages
            ema_12 = hist['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
            ema_26 = hist['Close'].ewm(span=26, adjust=False).mean().iloc[-1]
            
            # Calculate RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
            
            # Calculate MACD
            macd = ema_12 - ema_26
            signal = hist['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
            macd_histogram = macd - signal
            
            # Calculate Bollinger Bands
            sma_20_bb = hist['Close'].rolling(window=20).mean()
            std_20 = hist['Close'].rolling(window=20).std()
            upper_band = sma_20_bb + (std_20 * 2)
            lower_band = sma_20_bb - (std_20 * 2)
            
            # Calculate Average True Range (ATR)
            high_low = hist['High'] - hist['Low']
            high_close = np.abs(hist['High'] - hist['Close'].shift())
            low_close = np.abs(hist['Low'] - hist['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Calculate On-Balance Volume (OBV)
            obv = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum().iloc[-1]
            
            # Calculate Stochastic Oscillator
            low_min = hist['Low'].rolling(window=14).min()
            high_max = hist['High'].rolling(window=14).max()
            k = 100 * ((hist['Close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=3).mean()
            stoch_k = k.iloc[-1]
            stoch_d = d.iloc[-1]
            
            # Calculate Average Directional Index (ADX)
            plus_dm = hist['High'].diff()
            minus_dm = hist['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = true_range
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean().iloc[-1]
            
            # Calculate price momentum
            momentum_1m = ((current_price / hist['Close'].iloc[-22]) - 1) * 100
            momentum_3m = ((current_price / hist['Close'].iloc[-66]) - 1) * 100
            momentum_6m = ((current_price / hist['Close'].iloc[0]) - 1) * 100
            
            # Calculate volume trend
            volume_sma = hist['Volume'].rolling(window=20).mean().iloc[-1]
            volume_trend = (hist['Volume'].iloc[-1] / volume_sma - 1) * 100
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'rsi': rsi,
                'price_vs_sma20': (current_price / sma_20 - 1) * 100,
                'price_vs_sma50': (current_price / sma_50 - 1) * 100,
                'price_vs_sma200': (current_price / sma_200 - 1) * 100,
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': macd_histogram,
                'bollinger_upper': upper_band.iloc[-1],
                'bollinger_lower': lower_band.iloc[-1],
                'bollinger_position': (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) * 100,
                'atr': atr,
                'atr_percent': (atr / current_price) * 100,
                'obv': obv,
                'obv_trend': (obv / hist['Volume'].rolling(window=20).mean().iloc[-1] - 1) * 100,
                'stochastic_k': stoch_k,
                'stochastic_d': stoch_d,
                'adx': adx,
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'momentum_6m': momentum_6m,
                'volume_trend': volume_trend
            }
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment for {ticker}: {str(e)}")
            return {}

    def analyze_news_sentiment(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment from news content"""
        try:
            from news_analyst.news_analyzer import NewsAnalyzer
            
            news_analyzer = NewsAnalyzer()
            title = str(news_item.get('title', ''))
            description = str(news_item.get('description', ''))
            
            # Get sentiment from title
            title_sentiment = news_analyzer.analyze_sentiment(title)
            
            # Get sentiment from description if available
            desc_sentiment = news_analyzer.analyze_sentiment(description) if description else 'neutral'
            
            # Extract entities
            entities = news_analyzer.extract_entities(title)
            
            return {
                'title_sentiment': title_sentiment,
                'description_sentiment': desc_sentiment,
                'overall_sentiment': self._combine_sentiments(title_sentiment, desc_sentiment),
                'entities': entities
            }
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {
                'title_sentiment': 'neutral',
                'description_sentiment': 'neutral',
                'overall_sentiment': 'neutral',
                'entities': []
            }
        
    def _combine_sentiments(self, sentiment1: str, sentiment2: str) -> str:
        """Combine two sentiment scores"""
        if sentiment1 == sentiment2:
            return sentiment1
        
        # If one is neutral, return the other
        if sentiment1 == 'neutral':
            return sentiment2
        if sentiment2 == 'neutral':
            return sentiment1
        
        # If they're different, return neutral
        return 'neutral'

    def analyze_opportunity(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an investment opportunity and provide detailed research insights.
        
        Args:
            opportunity: Dictionary containing opportunity data
            
        Returns:
            Dictionary containing research analysis and recommendations
        """
        try:
            # Extract key metrics
            ticker = opportunity['ticker']
            price = opportunity['price']
            market_cap = opportunity['market_cap']
            
            # Calculate position sizing based on market cap and volatility
            position_size = self._calculate_position_sizing(opportunity)
            
            # Calculate risk parameters
            risk_params = self._calculate_risk_parameters(opportunity)
            
            # Analyze fundamental metrics
            fundamental_analysis = self._analyze_fundamentals(opportunity)
            
            # Analyze technical indicators
            technical_analysis = self._analyze_technicals(opportunity)
            
            # Analyze market sentiment
            sentiment_analysis = self._analyze_sentiment(opportunity)
            
            # Calculate risk-adjusted return metrics
            risk_metrics = self._calculate_risk_metrics(ticker)
            risk_assessment = self._evaluate_risk_adjusted_returns(risk_metrics)
            
            # Generate research report
            research_report = {
                'ticker': ticker,
                'name': opportunity['name'],
                'timestamp': get_current_time(),
                'current_price': price,
                'market_cap': market_cap,
                'industry': opportunity['industry'],
                'sector': opportunity['sector'],
                'position_size': position_size,
                'risk_parameters': risk_params,
                'fundamental_analysis': fundamental_analysis,
                'technical_analysis': technical_analysis,
                'sentiment_analysis': sentiment_analysis,
                'risk_assessment': risk_assessment,
                'recommendation': self._generate_recommendation(
                    fundamental_analysis,
                    technical_analysis,
                    sentiment_analysis,
                    risk_assessment
                ),
                'rationale': self._generate_rationale(
                    fundamental_analysis,
                    technical_analysis,
                    sentiment_analysis,
                    risk_assessment
                )
            }
            
            return research_report
            
        except Exception as e:
            self.logger.error(f"Error analyzing opportunity {opportunity.get('ticker', 'unknown')}: {str(e)}")
            return {}

    def _analyze_fundamentals(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamental metrics of the opportunity."""
        try:
            # Extract fundamental metrics
            pe_ratio = opportunity.get('pe_ratio')
            price_to_book = opportunity.get('price_to_book')
            debt_to_equity = opportunity.get('debt_to_equity')
            profit_margin = opportunity.get('profit_margin')
            revenue_growth_5y = opportunity.get('revenue_growth_5y')
            sales_growth_q = opportunity.get('sales_growth_q')
            sales_growth_a = opportunity.get('sales_growth_a')
            
            # Analyze valuation metrics
            valuation_analysis = {
                'pe_ratio': {
                    'value': pe_ratio,
                    'status': 'Undervalued' if pe_ratio and pe_ratio < 15 else 'Fair' if pe_ratio and pe_ratio < 25 else 'Overvalued',
                    'comment': f"P/E ratio of {pe_ratio:.2f} suggests {'attractive valuation' if pe_ratio and pe_ratio < 15 else 'fair valuation' if pe_ratio and pe_ratio < 25 else 'premium valuation'}"
                },
                'price_to_book': {
                    'value': price_to_book,
                    'status': 'Undervalued' if price_to_book and price_to_book < 1.5 else 'Fair' if price_to_book and price_to_book < 3 else 'Overvalued',
                    'comment': f"P/B ratio of {price_to_book:.2f} indicates {'potential value' if price_to_book and price_to_book < 1.5 else 'fair value' if price_to_book and price_to_book < 3 else 'premium value'}"
                }
            }
            
            # Analyze financial health
            financial_health = {
                'debt_to_equity': {
                    'value': debt_to_equity,
                    'status': 'Healthy' if debt_to_equity and debt_to_equity < 1.5 else 'Moderate' if debt_to_equity and debt_to_equity < 2.5 else 'High',
                    'comment': f"Debt/Equity ratio of {debt_to_equity:.2f} suggests {'strong financial position' if debt_to_equity and debt_to_equity < 1.5 else 'moderate leverage' if debt_to_equity and debt_to_equity < 2.5 else 'high leverage'}"
                },
                'profit_margin': {
                    'value': profit_margin,
                    'status': 'Strong' if profit_margin and profit_margin > 15 else 'Average' if profit_margin and profit_margin > 8 else 'Weak',
                    'comment': f"Profit margin of {profit_margin:.1f}% indicates {'strong profitability' if profit_margin and profit_margin > 15 else 'average profitability' if profit_margin and profit_margin > 8 else 'weak profitability'}"
                }
            }
            
            # Analyze growth metrics
            growth_analysis = {
                'revenue_growth_5y': {
                    'value': revenue_growth_5y,
                    'status': 'Strong' if revenue_growth_5y and revenue_growth_5y > 20 else 'Moderate' if revenue_growth_5y and revenue_growth_5y > 10 else 'Weak',
                    'comment': f"5-year revenue growth of {revenue_growth_5y:.1f}% shows {'strong growth trajectory' if revenue_growth_5y and revenue_growth_5y > 20 else 'moderate growth' if revenue_growth_5y and revenue_growth_5y > 10 else 'weak growth'}"
                },
                'sales_growth': {
                    'quarterly': sales_growth_q,
                    'annual': sales_growth_a,
                    'status': 'Accelerating' if sales_growth_q and sales_growth_a and sales_growth_q > sales_growth_a else 'Decelerating' if sales_growth_q and sales_growth_a and sales_growth_q < sales_growth_a else 'Stable',
                    'comment': f"Sales growth {'accelerating' if sales_growth_q and sales_growth_a and sales_growth_q > sales_growth_a else 'decelerating' if sales_growth_q and sales_growth_a and sales_growth_q < sales_growth_a else 'stable'} (Q: {sales_growth_q:.1f}%, A: {sales_growth_a:.1f}%)"
                }
            }
            
            return {
                'valuation': valuation_analysis,
                'financial_health': financial_health,
                'growth': growth_analysis,
                'overall_assessment': self._assess_fundamentals(valuation_analysis, financial_health, growth_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing fundamentals: {str(e)}")
            return {}

    def _assess_fundamentals(self, valuation: Dict[str, Any], financial_health: Dict[str, Any], growth: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall fundamental strength based on valuation, financial health, and growth metrics."""
        try:
            # Count positive and negative factors
            positive_factors = []
            negative_factors = []
            neutral_factors = []
            
            # Assess valuation
            for metric in valuation.values():
                status = metric.get('status', '')
                if status == 'Undervalued':
                    positive_factors.append(metric['comment'])
                elif status == 'Overvalued':
                    negative_factors.append(metric['comment'])
                else:
                    neutral_factors.append(metric['comment'])
            
            # Assess financial health
            for metric in financial_health.values():
                status = metric.get('status', '')
                if status in ['Healthy', 'Strong']:
                    positive_factors.append(metric['comment'])
                elif status in ['High', 'Weak']:
                    negative_factors.append(metric['comment'])
                else:
                    neutral_factors.append(metric['comment'])
            
            # Assess growth
            for metric in growth.values():
                status = metric.get('status', '')
                if status in ['Strong', 'Accelerating']:
                    positive_factors.append(metric['comment'])
                elif status in ['Weak', 'Decelerating']:
                    negative_factors.append(metric['comment'])
                else:
                    neutral_factors.append(metric['comment'])
            
            # Determine overall assessment
            if len(positive_factors) > len(negative_factors) + len(neutral_factors):
                strength = 'Strong'
            elif len(negative_factors) > len(positive_factors) + len(neutral_factors):
                strength = 'Weak'
            else:
                strength = 'Moderate'
            
            return {
                'strength': strength,
                'positive_factors': positive_factors,
                'negative_factors': negative_factors,
                'neutral_factors': neutral_factors,
                'summary': f"Overall fundamental analysis shows {strength.lower()} strength with {len(positive_factors)} positive, {len(negative_factors)} negative, and {len(neutral_factors)} neutral factors."
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing fundamentals: {str(e)}")
            return {
                'strength': 'Unknown',
                'positive_factors': [],
                'negative_factors': [],
                'neutral_factors': [],
                'summary': "Unable to assess fundamentals due to insufficient data."
            }

    def _analyze_technicals(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical indicators of the opportunity."""
        try:
            # Extract technical metrics
            price = opportunity['price']
            price_change_50d = opportunity.get('price_change_50d')
            price_change_200d = opportunity.get('price_change_200d')
            volume_change_50d = opportunity.get('volume_change_50d')
            volume_change_200d = opportunity.get('volume_change_200d')
            macd = opportunity.get('macd')
            rsi = opportunity.get('rsi')
            
            # Analyze price momentum
            momentum_analysis = {
                'short_term': {
                    'change': price_change_50d,
                    'status': 'Bullish' if price_change_50d and price_change_50d > 5 else 'Bearish' if price_change_50d and price_change_50d < -5 else 'Neutral',
                    'comment': f"50-day price change of {price_change_50d:.1f}% indicates {'strong upward momentum' if price_change_50d and price_change_50d > 5 else 'downward pressure' if price_change_50d and price_change_50d < -5 else 'sideways movement'}"
                },
                'long_term': {
                    'change': price_change_200d,
                    'status': 'Bullish' if price_change_200d and price_change_200d > 10 else 'Bearish' if price_change_200d and price_change_200d < -10 else 'Neutral',
                    'comment': f"200-day price change of {price_change_200d:.1f}% shows {'strong long-term uptrend' if price_change_200d and price_change_200d > 10 else 'long-term downtrend' if price_change_200d and price_change_200d < -10 else 'long-term consolidation'}"
                }
            }
            
            # Analyze volume trends
            volume_analysis = {
                'short_term': {
                    'change': volume_change_50d,
                    'status': 'High' if volume_change_50d and volume_change_50d > 50 else 'Average' if volume_change_50d and volume_change_50d > 0 else 'Low',
                    'comment': f"50-day volume change of {volume_change_50d:.1f}% indicates {'increasing trading interest' if volume_change_50d and volume_change_50d > 50 else 'normal trading activity' if volume_change_50d and volume_change_50d > 0 else 'declining interest'}"
                },
                'long_term': {
                    'change': volume_change_200d,
                    'status': 'High' if volume_change_200d and volume_change_200d > 30 else 'Average' if volume_change_200d and volume_change_200d > 0 else 'Low',
                    'comment': f"200-day volume change of {volume_change_200d:.1f}% shows {'sustained trading interest' if volume_change_200d and volume_change_200d > 30 else 'stable trading activity' if volume_change_200d and volume_change_200d > 0 else 'declining trading activity'}"
                }
            }
            
            # Analyze technical indicators
            indicator_analysis = {
                'macd': {
                    'value': macd,
                    'status': 'Bullish' if macd and macd > 0 else 'Bearish',
                    'comment': f"MACD of {macd:.2f} suggests {'bullish momentum' if macd and macd > 0 else 'bearish momentum'}"
                },
                'rsi': {
                    'value': rsi,
                    'status': 'Overbought' if rsi and rsi > 70 else 'Oversold' if rsi and rsi < 30 else 'Neutral',
                    'comment': f"RSI of {rsi:.1f} indicates {'overbought conditions' if rsi and rsi > 70 else 'oversold conditions' if rsi and rsi < 30 else 'neutral conditions'}"
                }
            }
            
            return {
                'momentum': momentum_analysis,
                'volume': volume_analysis,
                'indicators': indicator_analysis,
                'overall_assessment': self._assess_technicals(momentum_analysis, volume_analysis, indicator_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing technicals: {str(e)}")
            return {}

    def _analyze_sentiment(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment indicators."""
        try:
            # Extract sentiment metrics
            analyst_rating = opportunity.get('analyst_rating')
            mean_target = opportunity.get('mean_target')
            insider_ownership = opportunity.get('insider_ownership')
            institutional_ownership = opportunity.get('institutional_ownership')
            current_price = opportunity.get('price')
            
            # Analyze analyst sentiment
            analyst_sentiment = {
                'rating': {
                    'value': analyst_rating,
                    'status': 'Bullish' if 'BUY' in str(analyst_rating).upper() else 'Bearish' if 'SELL' in str(analyst_rating).upper() else 'Neutral',
                    'comment': f"Analyst rating of {analyst_rating} indicates {'positive outlook' if 'BUY' in str(analyst_rating).upper() else 'negative outlook' if 'SELL' in str(analyst_rating).upper() else 'neutral outlook'}"
                },
                'price_target': {
                    'value': mean_target,
                    'upside': ((mean_target / current_price - 1) * 100) if mean_target and current_price else None,
                    'status': 'Bullish' if mean_target and current_price and mean_target > current_price * 1.1 else 'Bearish' if mean_target and current_price and mean_target < current_price * 0.9 else 'Neutral',
                    'comment': f"Mean price target of ${mean_target:.2f} suggests {'significant upside' if mean_target and current_price and mean_target > current_price * 1.1 else 'potential downside' if mean_target and current_price and mean_target < current_price * 0.9 else 'limited price movement'}"
                }
            }
            
            # Analyze ownership sentiment
            ownership_sentiment = {
                'insider': {
                    'value': insider_ownership,
                    'status': 'High' if insider_ownership and insider_ownership > 20 else 'Average' if insider_ownership and insider_ownership > 10 else 'Low',
                    'comment': f"Insider ownership of {insider_ownership:.1f}% indicates {'strong insider confidence' if insider_ownership and insider_ownership > 20 else 'moderate insider interest' if insider_ownership and insider_ownership > 10 else 'limited insider involvement'}"
                },
                'institutional': {
                    'value': institutional_ownership,
                    'status': 'High' if institutional_ownership and institutional_ownership > 70 else 'Average' if institutional_ownership and institutional_ownership > 50 else 'Low',
                    'comment': f"Institutional ownership of {institutional_ownership:.1f}% shows {'strong institutional support' if institutional_ownership and institutional_ownership > 70 else 'moderate institutional interest' if institutional_ownership and institutional_ownership > 50 else 'limited institutional involvement'}"
                }
            }
            
            return {
                'analyst': analyst_sentiment,
                'ownership': ownership_sentiment,
                'overall_assessment': self._assess_sentiment(analyst_sentiment, ownership_sentiment)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {}

    def _generate_recommendation(self, fundamental_analysis: Dict[str, Any], technical_scores: Dict[str, Any], sentiment_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendation based on analysis"""
        # Enhanced scoring system
        fundamental_score = self._calculate_fundamental_score(fundamental_analysis)
        technical_score = self._calculate_technical_score(technical_scores)
        
        # Use the opportunity score from InvestmentAnalyzer
        opportunity_score = sentiment_analysis.get('opportunity', {}).get('score', 0.0)
        
        # Calculate overall confidence score with adjusted weights
        confidence_score = (
            fundamental_score * 0.2 +  # Reduced weight for fundamental score
            technical_score * 0.3 +    # Maintained weight for technical score
            opportunity_score * 0.5    # Increased weight for opportunity score
        )
        
        # Determine action based on confidence score and price change
        price_change = sentiment_analysis.get('opportunity', {}).get('price_change', 0)
        
        if confidence_score >= 0.3:  # Lowered from 0.4
            if price_change > 0:
                action = 'STRONG_BUY'
            else:
                action = 'STRONG_SELL'
        elif confidence_score >= 0.2:  # Lowered from 0.3
            if price_change > 0:
                action = 'BUY'
            else:
                action = 'SELL'
        elif confidence_score >= 0.1:  # Kept the same
            if price_change > 0:
                action = 'SELL'
            else:
                action = 'BUY'
        else:
            if price_change > 0:
                action = 'STRONG_SELL'
            else:
                action = 'STRONG_BUY'
            
        return {
            'action': action,
            'confidence_score': confidence_score,
            'fundamental_score': fundamental_score,
            'technical_score': technical_score,
            'opportunity_score': opportunity_score,
            'risk_assessment': risk_assessment
        }
        
    def _calculate_fundamental_score(self, fundamental_analysis: Dict[str, Any]) -> float:
        """Calculate fundamental analysis score"""
        try:
            # Extract key metrics
            current_ratio = fundamental_analysis.get('current_ratio', 0)
            debt_to_equity = fundamental_analysis.get('debt_to_equity', 0)
            profit_margin = fundamental_analysis.get('profit_margin', 0)
            revenue_growth = fundamental_analysis.get('revenue_growth', 0)
            earnings_growth = fundamental_analysis.get('earnings_growth', 0)
            
            # Score current ratio (ideal > 1.5)
            current_ratio_score = min(current_ratio / 1.5, 1.0) if current_ratio > 0 else 0
            
            # Score debt to equity (ideal < 1.0)
            debt_to_equity_score = max(1 - (debt_to_equity / 2), 0) if debt_to_equity > 0 else 1
            
            # Score profit margin (ideal > 15%)
            profit_margin_score = min(profit_margin / 15, 1.0) if profit_margin > 0 else 0
            
            # Score growth metrics (ideal > 10%)
            revenue_growth_score = min(revenue_growth / 10, 1.0) if revenue_growth > 0 else 0
            earnings_growth_score = min(earnings_growth / 10, 1.0) if earnings_growth > 0 else 0
            
            # Calculate weighted average
            fundamental_score = (
                current_ratio_score * 0.2 +
                debt_to_equity_score * 0.2 +
                profit_margin_score * 0.2 +
                revenue_growth_score * 0.2 +
                earnings_growth_score * 0.2
            )
            
            return max(min(fundamental_score, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating fundamental score: {str(e)}")
            return 0.0
            
    def _calculate_technical_score(self, technical_scores: Dict[str, Any]) -> float:
        """Calculate technical analysis score"""
        try:
            # Extract key metrics
            rsi = technical_scores.get('rsi', 50)
            price_vs_sma20 = technical_scores.get('price_vs_sma20', 0)
            price_vs_sma50 = technical_scores.get('price_vs_sma50', 0)
            price_vs_sma200 = technical_scores.get('price_vs_sma200', 0)
            macd_histogram = technical_scores.get('macd_histogram', 0)
            bollinger_position = technical_scores.get('bollinger_position', 50)
            stochastic_k = technical_scores.get('stochastic_k', 50)
            stochastic_d = technical_scores.get('stochastic_d', 50)
            adx = technical_scores.get('adx', 0)
            momentum_1m = technical_scores.get('momentum_1m', 0)
            momentum_3m = technical_scores.get('momentum_3m', 0)
            momentum_6m = technical_scores.get('momentum_6m', 0)
            volume_trend = technical_scores.get('volume_trend', 0)
            
            # Score RSI (ideal between 40-60)
            rsi_score = 1 - abs(rsi - 50) / 50
            
            # Score moving averages
            ma_score = (
                (1 if price_vs_sma20 > 0 else 0) * 0.4 +
                (1 if price_vs_sma50 > 0 else 0) * 0.3 +
                (1 if price_vs_sma200 > 0 else 0) * 0.3
            )
            
            # Score MACD
            macd_score = 1 if macd_histogram > 0 else 0
            
            # Score Bollinger Bands
            bb_score = 1 - abs(bollinger_position - 50) / 50
            
            # Score Stochastic
            stoch_score = 1 - abs(stochastic_k - 50) / 50
            
            # Score ADX (trend strength)
            adx_score = min(adx / 25, 1.0)
            
            # Score momentum
            momentum_score = (
                (1 if momentum_1m > 0 else 0) * 0.4 +
                (1 if momentum_3m > 0 else 0) * 0.3 +
                (1 if momentum_6m > 0 else 0) * 0.3
            )
            
            # Score volume trend
            volume_score = 1 if volume_trend > 0 else 0
            
            # Calculate weighted average
            technical_score = (
                rsi_score * 0.15 +
                ma_score * 0.2 +
                macd_score * 0.1 +
                bb_score * 0.1 +
                stoch_score * 0.1 +
                adx_score * 0.1 +
                momentum_score * 0.15 +
                volume_score * 0.1
            )
            
            return max(min(technical_score, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score: {str(e)}")
            return 0.0

    def _calculate_time_horizon(self, fundamental_analysis: Dict[str, Any], technical_scores: Dict[str, Any], market_sentiment: Dict[str, Any], opportunity: Dict[str, Any]) -> str:
        """Determine the time horizon for the recommendation"""
        # Determine time horizon based on opportunity score and market cap
        opportunity_score = opportunity.get('score', 0.0)
        market_cap = opportunity.get('market_cap', 0)
        
        if opportunity_score >= 0.7 and market_cap >= 10_000_000_000:  # High score and large cap
            return "long_term"
        elif opportunity_score >= 0.5 or market_cap >= 1_000_000_000:  # Moderate score or mid cap
            return "medium_term"
        else:
            return "short_term"

    def _calculate_risk_parameters(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk parameters for a trade opportunity."""
        try:
            # Get market cap in billions
            market_cap = opportunity['market_cap'] / 1_000_000_000
            
            # Base risk parameters
            if market_cap >= 200:  # Mega cap
                risk_level = "Low"
                max_loss_percent = 2.0
            elif market_cap >= 10:  # Large cap
                risk_level = "Low"
                max_loss_percent = 2.5
            elif market_cap >= 1:  # Mid cap (updated threshold)
                risk_level = "Medium"
                max_loss_percent = 3.0
            else:  # Small cap (below $1B)
                risk_level = "High"
                max_loss_percent = 4.0
                
            # Adjust stop loss based on volatility (using beta if available)
            beta = opportunity.get('beta', 1.0)
            stop_loss_percent = max_loss_percent * (1 + (beta - 1) * 0.5)
            
            # Calculate take profit level (risk:reward of 1:2)
            take_profit_percent = stop_loss_percent * 2
            
            return {
                'stop_loss_percent': stop_loss_percent,
                'take_profit_percent': take_profit_percent,
                'risk_level': risk_level,
                'max_loss_percent': max_loss_percent,
                'trailing_stop': True  # Enable trailing stop by default
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {str(e)}")
            return {
                'stop_loss_percent': 2.0,
                'take_profit_percent': 4.0,
                'risk_level': "Medium",
                'max_loss_percent': 2.0,
                'trailing_stop': True
            }
            
    def _calculate_position_sizing(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position sizing for a trade opportunity."""
        try:
            # Get market cap in billions
            market_cap = opportunity['market_cap'] / 1_000_000_000
            
            # Base position size based on market cap
            if market_cap >= 200:  # Mega cap
                base_size = 0.10  # Up to 10% of portfolio
            elif market_cap >= 10:  # Large cap
                base_size = 0.05  # Up to 5% of portfolio
            elif market_cap >= 1:  # Mid cap (updated threshold)
                base_size = 0.03  # Up to 3% of portfolio
            else:  # Small cap (below $1B)
                base_size = 0.01  # Up to 1% of portfolio (reduced from 2%)
                
            # Adjust for score
            score = opportunity.get('score', 0.5)
            adjusted_size = base_size * score
            
            # Adjust for volatility
            beta = opportunity.get('beta', 1.0)
            if beta > 1.5:
                adjusted_size *= 0.8  # Reduce position for high volatility
            elif beta < 0.5:
                adjusted_size *= 1.2  # Increase position for low volatility
                
            # Ensure minimum and maximum sizes
            min_size = 0.01  # 1% minimum
            max_size = 0.15  # 15% maximum
            final_size = max(min_size, min(adjusted_size, max_size))
            
            return {
                'position_size': round(final_size, 4),
                'base_size': base_size,
                'score_adjustment': score,
                'volatility_adjustment': beta,
                'min_size': min_size,
                'max_size': max_size
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizing: {str(e)}")
            return {
                'position_size': 0.02,
                'base_size': 0.02,
                'score_adjustment': 1.0,
                'volatility_adjustment': 1.0,
                'min_size': 0.01,
                'max_size': 0.15
            }

    def _assess_technicals(self, momentum_analysis: Dict[str, Any], volume_analysis: Dict[str, Any], indicator_analysis: Dict[str, Any]) -> str:
        """Assess overall technical analysis"""
        # Combine individual assessments
        overall_assessment = {
            'momentum': momentum_analysis['overall_assessment'],
            'volume': volume_analysis['overall_assessment'],
            'indicators': indicator_analysis['overall_assessment'],
            'overall_score': (
                momentum_analysis['overall_assessment'] * 0.3 +
                volume_analysis['overall_assessment'] * 0.3 +
                indicator_analysis['overall_assessment'] * 0.4
            )
        }
        
        # Determine overall assessment based on score
        if overall_assessment['overall_score'] >= 0.7:
            return "Strong"
        elif overall_assessment['overall_score'] >= 0.5:
            return "Moderate"
        else:
            return "Weak"

    def _assess_sentiment(self, analyst_sentiment: Dict[str, Any], ownership_sentiment: Dict[str, Any]) -> str:
        """Assess overall sentiment analysis"""
        # Combine individual assessments
        overall_assessment = {
            'analyst': analyst_sentiment['overall_assessment'],
            'ownership': ownership_sentiment['overall_assessment'],
            'overall_score': (
                analyst_sentiment['overall_assessment'] * 0.5 +
                ownership_sentiment['overall_assessment'] * 0.5
            )
        }
        
        # Determine overall assessment based on score
        if overall_assessment['overall_score'] >= 0.7:
            return "Strong"
        elif overall_assessment['overall_score'] >= 0.5:
            return "Moderate"
        else:
            return "Weak"

    def _generate_rationale(self, fundamental_analysis: Dict[str, Any], technical_scores: Dict[str, Any], sentiment_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> str:
        """Generate a rationale for the trade recommendation."""
        try:
            ticker = fundamental_analysis.get('ticker', 'Unknown')
            recommendation = sentiment_analysis.get('recommendation', {})
            action = recommendation.get('action', 'Unknown')
            confidence_score = recommendation.get('confidence_score', 0)
            
            # Get company info
            company_info = self.get_company_info(ticker)
            sector = company_info.get('sector', 'Unknown')
            industry = company_info.get('industry', 'Unknown')
            
            # Get opportunity data
            opportunity = sentiment_analysis.get('opportunity', {})
            price_change = opportunity.get('price_change', 0)
            volume_change = opportunity.get('volume_change', 0)
            market_cap = opportunity.get('market_cap', 0)
            
            # Format market cap
            if market_cap >= 1_000_000_000_000:
                market_cap_str = f"${market_cap/1_000_000_000_000:.2f}T"
            elif market_cap >= 1_000_000_000:
                market_cap_str = f"${market_cap/1_000_000_000:.2f}B"
            elif market_cap >= 1_000_000:
                market_cap_str = f"${market_cap/1_000_000:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.2f}"
                
            # Generate rationale
            rationale = f"{action} recommendation for {ticker} ({sector} - {industry}) with {confidence_score:.1%} confidence. "
            
            if price_change > 0:
                rationale += f"Showing positive price momentum of {price_change:.1%}. "
            else:
                rationale += f"Showing negative price momentum of {price_change:.1%}. "
                
            rationale += f"Market cap: {market_cap_str}. "
            
            if volume_change > 0:
                rationale += f"Volume is {volume_change:.1%} above average. "
            else:
                rationale += f"Volume is {abs(volume_change):.1%} below average. "
                
            # Add sentiment analysis
            sentiment_score = sentiment_analysis.get('overall_sentiment', 'Neutral')
            rationale += f"Sentiment: {sentiment_score}. "
            
            # Add risk assessment
            risk_level = risk_assessment.get('risk_level', 'Unknown')
            rationale += f"Risk level: {risk_level}. "
            
            return rationale
            
        except Exception as e:
            self.logger.error(f"Error generating rationale: {str(e)}")
            return "Insufficient data to generate rationale."

    def _validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker is tradeable and represents a real company"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we can get basic company info
            if not info:
                return False
                
            # Check if it has a market price
            if not info.get('regularMarketPrice'):
                return False
                
            # Check if it has a company name (to filter out non-company tickers)
            if not info.get('longName') or not info.get('shortName'):
                return False
                
            # Check if it's a real company with market cap
            if not info.get('marketCap') or info.get('marketCap') < 1000000:  # At least $1M market cap
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating ticker {ticker}: {str(e)}")
            return False

    def _calculate_technical_scores(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Calculate technical indicators and scores."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            
            if hist.empty:
                return None
            
            # Calculate RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate Moving Averages
            sma20 = hist['Close'].rolling(window=20).mean()
            sma50 = hist['Close'].rolling(window=50).mean()
            sma200 = hist['Close'].rolling(window=200).mean()
            
            current_price = hist['Close'].iloc[-1]
            price_vs_sma20 = ((current_price - sma20.iloc[-1]) / sma20.iloc[-1]) * 100
            price_vs_sma50 = ((current_price - sma50.iloc[-1]) / sma50.iloc[-1]) * 100
            price_vs_sma200 = ((current_price - sma200.iloc[-1]) / sma200.iloc[-1]) * 100
            
            # Calculate MACD
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            
            # Calculate Bollinger Bands
            sma20 = hist['Close'].rolling(window=20).mean()
            std20 = hist['Close'].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            
            bollinger_position = ((current_price - lower_band.iloc[-1]) / 
                                (upper_band.iloc[-1] - lower_band.iloc[-1])) * 100
            
            # Calculate Stochastic Oscillator
            low_14 = hist['Low'].rolling(window=14).min()
            high_14 = hist['High'].rolling(window=14).max()
            k = ((hist['Close'] - low_14) / (high_14 - low_14)) * 100
            stoch_k = k.rolling(window=3).mean()
            stoch_d = stoch_k.rolling(window=3).mean()
            
            # Calculate ADX
            plus_dm = hist['High'].diff()
            minus_dm = hist['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr1 = hist['High'] - hist['Low']
            tr2 = abs(hist['High'] - hist['Close'].shift())
            tr3 = abs(hist['Low'] - hist['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = abs(100 * (minus_dm.rolling(window=14).mean() / atr))
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            # Calculate Momentum
            momentum_1m = ((current_price - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]) * 100
            momentum_3m = ((current_price - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60]) * 100
            momentum_6m = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            
            # Calculate Volume Trend
            volume_sma = hist['Volume'].rolling(window=20).mean()
            volume_trend = ((hist['Volume'].iloc[-1] - volume_sma.iloc[-1]) / volume_sma.iloc[-1]) * 100
            
            return {
                'rsi': rsi.iloc[-1],
                'price_vs_sma20': price_vs_sma20,
                'price_vs_sma50': price_vs_sma50,
                'price_vs_sma200': price_vs_sma200,
                'macd_histogram': macd_histogram.iloc[-1],
                'bollinger_position': bollinger_position,
                'stochastic_k': stoch_k.iloc[-1],
                'stochastic_d': stoch_d.iloc[-1],
                'adx': adx.iloc[-1],
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'momentum_6m': momentum_6m,
                'volume_trend': volume_trend,
                'atr': atr.iloc[-1],
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical scores for {ticker}: {str(e)}")
            return None

    def process_flagged_opportunities(self, opportunities: List[Dict]) -> None:
        """Process flagged opportunities from the InvestmentAnalyzer."""
        try:
            if not opportunities:
                self.logger.warning("No opportunities to process")
                return
                
            self.logger.info(f"Processing {len(opportunities)} opportunities")
            
            # Save opportunities to file
            save_to_json(opportunities, self.opportunities_file)
            
            # Process each opportunity
            analysis_results = []
            for opportunity in opportunities:
                try:
                    # Get company info
                    company_info = self._get_company_info(opportunity['ticker'])
                    
                    # Analyze fundamentals
                    fundamental_analysis = self._analyze_fundamentals(opportunity)
                    
                    # Calculate risk parameters
                    risk_parameters = self._calculate_risk_parameters(opportunity)
                    
                    # Calculate position sizing
                    position_sizing = self._calculate_position_sizing(opportunity)
                    
                    # Combine all analysis
                    analysis = {
                        'ticker': opportunity['ticker'],
                        'opportunity': opportunity,
                        'company_info': company_info,
                        'fundamental_analysis': fundamental_analysis,
                        'risk_parameters': risk_parameters,
                        'position_sizing': position_sizing,
                        'timestamp': get_current_time().isoformat()
                    }
                    
                    analysis_results.append(analysis)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {opportunity['ticker']}: {str(e)}")
                    continue
            
            # Save analysis results
            if analysis_results:
                save_to_json(analysis_results, self.analysis_file)
                self.logger.info(f"Saved analysis results for {len(analysis_results)} opportunities")
            else:
                self.logger.warning("No analysis results to save")
            
        except Exception as e:
            self.logger.error(f"Error processing opportunities: {str(e)}")
            
    def _get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get company information from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', None),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'short_ratio': info.get('shortRatio', 0),
                'profit_margins': info.get('profitMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return {}
            
    def _calculate_confidence_score(self, analysis: Dict) -> float:
        """Calculate confidence score based on analysis results."""
        score = 0.0
        
        # Fundamental analysis (30% weight - reduced further)
        fundamental_score = 0.0
        if analysis.get('fundamental_analysis'):
            fundamental = analysis['fundamental_analysis']
            if fundamental.get('pe_ratio') and fundamental['pe_ratio'] < 60:  # Even more lenient PE ratio
                fundamental_score += 0.2
            if fundamental.get('pb_ratio') and fundamental['pb_ratio'] < 6:  # Even more lenient PB ratio
                fundamental_score += 0.2
        score += fundamental_score * 0.3
        
        # Technical analysis (40% weight - increased further)
        technical_score = 0.0
        if analysis.get('technical_analysis'):
            technical = analysis['technical_analysis']
            if technical.get('rsi') and 15 <= technical['rsi'] <= 85:  # Even wider RSI range
                technical_score += 0.15
            if technical.get('macd') and technical['macd'] > -1.5:  # Even more lenient MACD
                technical_score += 0.15
        score += technical_score * 0.4
        
        # Risk parameters (30% weight)
        risk_score = 0.0
        if analysis.get('risk_parameters'):
            risk = analysis['risk_parameters']
            if risk.get('volatility') and risk['volatility'] < 0.7:  # Even higher volatility threshold
                risk_score += 0.15
            if risk.get('beta') and 0.3 <= risk['beta'] <= 2.5:  # Even wider beta range
                risk_score += 0.15
        score += risk_score * 0.3
        
        return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1

    def _should_recommend(self, analysis: Dict) -> bool:
        """Determine if a stock should be recommended based on analysis."""
        confidence_score = self._calculate_confidence_score(analysis)
        return confidence_score >= 0.2  # Even lower confidence threshold from 0.25 to 0.2

    def _calculate_risk_metrics(self, symbol: str, timeframe: str = '1y') -> Dict[str, float]:
        """
        Calculate risk-adjusted return metrics for a given stock.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Time period for analysis (default: 1 year)
            
        Returns:
            Dictionary containing risk metrics
        """
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            stock_data = stock.history(period=timeframe)
            
            # Get market data (S&P 500 as benchmark)
            market_data = yf.download('^GSPC', period=timeframe)['Adj Close']
            
            # Calculate daily returns
            stock_returns = stock_data['Adj Close'].pct_change().dropna()
            market_returns = market_data.pct_change().dropna()
            
            # Align the data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            stock_returns = aligned_data.iloc[:, 0]
            market_returns = aligned_data.iloc[:, 1]
            
            # Calculate metrics
            # Beta
            beta = np.cov(stock_returns, market_returns)[0][1] / np.var(market_returns)
            
            # Alpha
            alpha = (stock_returns.mean() - self.risk_free_rate/252) - \
                   beta * (market_returns.mean() - self.risk_free_rate/252)
            alpha = alpha * 252  # Annualize alpha
            
            # Standard Deviation
            std_dev = stock_returns.std() * np.sqrt(252)  # Annualized
            
            # Sharpe Ratio
            excess_returns = stock_returns - self.risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / stock_returns.std()
            
            # Maximum Drawdown
            cummax = stock_data['Adj Close'].cummax()
            drawdown = (stock_data['Adj Close'] - cummax) / cummax
            max_drawdown = drawdown.min()
            
            return {
                'alpha': round(alpha, 4),
                'beta': round(beta, 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'std_dev': round(std_dev, 4),
                'max_drawdown': round(max_drawdown, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics for {symbol}: {str(e)}")
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'sharpe_ratio': 0.0,
                'std_dev': 0.0,
                'max_drawdown': 0.0
            }
    
    def _evaluate_risk_adjusted_returns(self, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate risk-adjusted return metrics against risk tolerance parameters.
        
        Args:
            risk_metrics: Dictionary of risk metrics
            
        Returns:
            Dictionary containing evaluation results and risk score
        """
        try:
            # Initialize scores
            scores = {
                'alpha_score': 1.0 if risk_metrics['alpha'] >= self.min_alpha else 0.0,
                'beta_score': 1.0 if risk_metrics['beta'] <= self.max_beta else 0.0,
                'sharpe_score': 1.0 if risk_metrics['sharpe_ratio'] >= self.min_sharpe_ratio else 0.0,
                'volatility_score': 1.0 if risk_metrics['std_dev'] <= 0.4 else 0.0,  # 40% max volatility
                'drawdown_score': 1.0 if risk_metrics['max_drawdown'] >= -0.3 else 0.0  # -30% max drawdown
            }
            
            # Calculate overall risk score (weighted average)
            weights = {
                'alpha_score': 0.3,
                'beta_score': 0.2,
                'sharpe_score': 0.3,
                'volatility_score': 0.1,
                'drawdown_score': 0.1
            }
            
            risk_score = sum(score * weights[metric] for metric, score in scores.items())
            
            # Generate risk assessment
            risk_assessment = {
                'risk_score': round(risk_score, 4),
                'risk_level': 'LOW' if risk_score >= 0.8 else 'MEDIUM' if risk_score >= 0.5 else 'HIGH',
                'metrics': risk_metrics,
                'scores': scores,
                'recommendation': 'ACCEPT' if risk_score >= 0.7 else 'REVIEW' if risk_score >= 0.4 else 'REJECT'
            }
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error evaluating risk-adjusted returns: {str(e)}")
            return {
                'risk_score': 0.0,
                'risk_level': 'HIGH',
                'metrics': risk_metrics,
                'scores': {},
                'recommendation': 'REJECT'
            }

if __name__ == "__main__":
    analyzer = ResearchAnalyzer()
    analyzer.process_flagged_opportunities()