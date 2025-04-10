"""
Portfolio Risk Analysis Engine

This module is the core engine that calculates risk scores for stocks and securities.
It provides a comprehensive risk assessment system that helps investors understand
the risk profile of their investments.

Key Features:
1. Risk Score Calculation (0-100 scale):
   - 0-25: Very low risk (stable, established companies)
   - 26-50: Moderate risk (growing companies with some volatility)
   - 51-75: High risk (volatile stocks, emerging companies)
   - 76-100: Very high risk (speculative investments)

2. Risk Factors Considered:
   - Market Risk:
     * Price volatility (how much the stock price moves)
     * Beta (how the stock moves compared to the market)
     * Maximum drawdown (worst price drop)
     * Upside vs downside volatility
   
   - Financial Risk:
     * Company size and market capitalization
     * Debt levels and financial health
     * Profitability metrics
     * Cash flow analysis
   
   - Sector Risk:
     * Industry-specific risks
     * Market position
     * Competitive landscape
   
   - Technical Risk:
     * Trading volume and liquidity
     * RSI (Relative Strength Index)
     * Price momentum
     * Volume trends

3. Data Sources:
   - Yahoo Finance API for market data
   - Company financial statements
   - Technical indicators
   - Market statistics

The analyzer uses a weighted scoring system where each risk factor contributes
to the final risk score based on its importance. This provides a balanced view
of both fundamental and technical risk factors.

Example Usage:
    analyzer = ResearchAnalyzer()
    risk_score = analyzer._calculate_risk_score(security_data)
"""

import os
import time
import random
import logging
import traceback
import json
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.common import setup_logging, get_current_time, save_to_json, load_from_json
from .rate_limiter import RateLimiter


class ResearchAnalyzer:
    """
    Analyzes high-potential securities identified by the InvestmentAnalyzer 
    to calculate a risk score (0-100) for each. Ranks securities by ascending 
    risk score for the Trade Analyst.
    """

    def __init__(self):
        """Initialize the ResearchAnalyzer."""
        self.logger = setup_logging('research_analyzer')
        # Use root-level data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        # Input file from InvestmentAnalyzer
        self.input_file = os.path.join(self.data_dir, 'high_potential_securities.json')
        # Output file for TradeAnalyst
        self.output_file = os.path.join(self.data_dir, 'risk_scored_securities.json')

        # Rate limiter for external API calls
        self.rate_limiter = RateLimiter(calls_per_second=6.0, max_retries=3, retry_delay=5.0)
        
        # Cache for fetched yfinance data to reduce API calls
        self.data_cache = {}

    def _fetch_yf_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches data from yfinance API with caching.
        Returns a dictionary containing 'info' and 'history' data.
        """
        yf_data = {'info': {}, 'history': pd.DataFrame()}
        
        # Check cache first
        cache_key = f"{ticker}_data"
        if cache_key in self.data_cache:
            self.logger.debug(f"Cache hit for {ticker}")
            return self.data_cache[cache_key]
        
        try:
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            
            # Fetch info data
            info_data = {}
            try:
                info_data = stock.info
            except Exception as e:
                self.logger.warning(f"Error fetching info data for {ticker}: {str(e)}")
            
            # Fetch history data
            hist_data = pd.DataFrame()
            try:
                hist_data = stock.history(period='1y')
            except Exception as e:
                self.logger.warning(f"Error fetching historical data for {ticker}: {str(e)}")
            
            # Update yf_data only if we have valid data
            if info_data:
                yf_data['info'] = info_data
            if not hist_data.empty:
                yf_data['history'] = hist_data
            
            # Cache the data
            self.data_cache[cache_key] = yf_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        
        return yf_data

    def _calculate_volatility_metrics(self, history: pd.DataFrame) -> Dict[str, Optional[float]]:
        """
        Calculates comprehensive volatility metrics from price history.
        Includes standard deviation, maximum drawdown, upside/downside volatility,
        Value at Risk (VaR), and Conditional Value at Risk (CVaR).
        """
        metrics = {
            'std_dev': None,
            'max_drawdown': None,
            'upside_vol': None,
            'downside_vol': None,
            'var_95': None,  # 95% Value at Risk
            'cvar_95': None,  # 95% Conditional VaR
            'volatility_trend': None  # Trend in volatility
        }
        
        try:
            if not history.empty and 'Close' in history.columns and len(history) > 1:
                daily_returns = history['Close'].pct_change().dropna()
                if not daily_returns.empty:
                    # Annualized standard deviation
                    metrics['std_dev'] = float(daily_returns.std() * np.sqrt(252))
                    
                    # Maximum drawdown
                    cummax = history['Close'].cummax()
                    drawdown = (history['Close'] - cummax) / cummax
                    metrics['max_drawdown'] = float(abs(drawdown.min()))
                    
                    # Upside and downside volatility
                    positive_returns = daily_returns[daily_returns > 0]
                    negative_returns = daily_returns[daily_returns < 0]
                    
                    metrics['upside_vol'] = float(positive_returns.std() * np.sqrt(252)) if not positive_returns.empty else 0.0
                    metrics['downside_vol'] = float(abs(negative_returns.std() * np.sqrt(252))) if not negative_returns.empty else 0.0
                    
                    # Value at Risk (VaR) at 95% confidence level
                    metrics['var_95'] = float(abs(np.percentile(daily_returns, 5)))
                    
                    # Conditional VaR (CVaR/Expected Shortfall)
                    cvar = abs(daily_returns[daily_returns <= -metrics['var_95']].mean())
                    metrics['cvar_95'] = float(cvar) if not np.isnan(cvar) else None
                    
                    # Volatility trend (using 30-day rolling volatility)
                    rolling_vol = daily_returns.rolling(window=30).std()
                    if len(rolling_vol) >= 30:
                        recent_vol = rolling_vol[-30:].mean()
                        past_vol = rolling_vol[-60:-30].mean() if len(rolling_vol) >= 60 else recent_vol
                        trend = (recent_vol / past_vol) - 1 if past_vol > 0 else 0
                        metrics['volatility_trend'] = float(trend)
                    else:
                        metrics['volatility_trend'] = 0.0
        except Exception as e:
            self.logger.warning(f"Could not calculate volatility metrics: {str(e)}")
                
        return metrics

    def _calculate_risk_score(self, security: Dict[str, Any]) -> int:
        """
        Calculates a risk score (0-100) for a single security.
        Higher score means higher risk.
        """
        ticker = security.get('Ticker')
        if not ticker:
            self.logger.warning("Security missing Ticker, cannot calculate risk.")
            return 75  # Assign higher risk if ticker is missing

        # Fetch required data from yfinance
        yf_data = self._fetch_yf_data(ticker)
        yf_info = yf_data.get('info', {})
        history = yf_data.get('history', pd.DataFrame())

        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(history)

        # Initialize risk factors
        risk_factors = {
            'volatility_risk': 0.0,
            'market_risk': 0.0,
            'financial_risk': 0.0,
            'sector_risk': 0.0,
            'technical_risk': 0.0
        }

        # Base risk for missing data
        missing_data_penalty = 0
        required_fields = ['Beta', 'Market Cap', 'Volume', 'rsi', 'Industry']
        for field in required_fields:
            if field not in security or not security[field]:
                missing_data_penalty += 10

        # Calculate volatility risk (25% weight)
        if volatility_metrics['std_dev'] is not None:
            risk_factors['volatility_risk'] = min(volatility_metrics['std_dev'] * 10, 12.5)
        if volatility_metrics['max_drawdown'] is not None:
            risk_factors['volatility_risk'] += min(volatility_metrics['max_drawdown'] * 100, 12.5)

        # Calculate market risk (20% weight)
        beta = security.get('Beta', 1.0)
        if 'Beta' in security:
            # Beta < 1 means lower volatility relative to market
            # Beta > 1 means higher volatility relative to market
            if beta < 1.0:
                beta_deviation = 1.0 - beta
                if beta_deviation <= 0.2:  # Slightly defensive (0.8-1.0)
                    risk_factors['market_risk'] = 5
                elif beta_deviation <= 0.5:  # Moderately defensive (0.5-0.8)
                    risk_factors['market_risk'] = 8
                else:  # Very defensive (< 0.5)
                    risk_factors['market_risk'] = 10
            else:
                beta_deviation = beta - 1.0
                if beta_deviation <= 0.2:  # Slightly aggressive (1.0-1.2)
                    risk_factors['market_risk'] = 10
                elif beta_deviation <= 0.5:  # Moderately aggressive (1.2-1.5)
                    risk_factors['market_risk'] = 15
                else:  # Very aggressive (> 1.5)
                    risk_factors['market_risk'] = 20
        else:
            risk_factors['market_risk'] = 15  # Default risk for missing beta

        # Calculate financial risk (20% weight)
        market_cap = security.get('Market Cap', 0)
        if market_cap < 1e9:  # Less than 1B
            risk_factors['financial_risk'] += 12
        elif market_cap < 10e9:  # Less than 10B
            risk_factors['financial_risk'] += 8
        elif market_cap < 100e9:  # Less than 100B
            risk_factors['financial_risk'] += 4

        # Add debt and profitability risk
        debt_to_equity = yf_info.get('debtToEquity', 0)
        profit_margins = yf_info.get('profitMargins', 0)
        if debt_to_equity > 2:
            risk_factors['financial_risk'] += 4
        if profit_margins < 0:
            risk_factors['financial_risk'] += 4

        # Calculate sector risk (20% weight)
        sector_risk_map = {
            'Technology': 20,
            'Healthcare': 16,
            'Financial Services': 14,
            'Consumer Cyclical': 12,
            'Communication Services': 10,
            'Industrials': 8,
            'Consumer Defensive': 6,
            'Utilities': 4,
            'Energy': 3,
            'Real Estate': 2,
            'Basic Materials': 1
        }
        industry = security.get('Industry', 'Unknown')
        risk_factors['sector_risk'] = sector_risk_map.get(industry, 15)  # Higher default risk for unknown industry

        # Calculate technical risk (15% weight)
        rsi = security.get('rsi', 0.5)
        if 'rsi' in security:
            # Lower risk for RSI close to 0.5, higher risk for extreme values
            rsi_deviation = abs(rsi - 0.5)
            if rsi_deviation <= 0.1:  # Close to neutral
                risk_factors['technical_risk'] += 2
            elif rsi_deviation <= 0.2:  # Moderately extreme
                risk_factors['technical_risk'] += 4
            else:  # Very extreme
                risk_factors['technical_risk'] += min(rsi_deviation * 25, 7.5)
        else:
            risk_factors['technical_risk'] += 5  # Default risk for missing RSI

        volume_change = security.get('volume_change_50d', 0)
        if 'volume_change_50d' in security:
            # Lower risk for small volume changes
            if abs(volume_change) <= 0.1:  # Small change
                risk_factors['technical_risk'] += 2
            elif abs(volume_change) <= 0.3:  # Moderate change
                risk_factors['technical_risk'] += 4
            else:  # Large change
                risk_factors['technical_risk'] += min(abs(volume_change) * 5, 7.5)
        else:
            risk_factors['technical_risk'] += 5  # Default risk for missing volume change

        # Calculate final risk score (0-100)
        total_risk = (
            risk_factors['volatility_risk'] +
            risk_factors['market_risk'] +
            risk_factors['financial_risk'] +
            risk_factors['sector_risk'] +
            risk_factors['technical_risk'] +
            missing_data_penalty  # Add penalty for missing data
        )

        # Ensure score is between 0 and 100
        final_score = min(max(int(total_risk), 0), 100)
        
        self.logger.debug(f"Risk score for {ticker}: {final_score}")
        return final_score

    def run_risk_analysis(self) -> Union[Dict[str, Any], bool]:
        """
        Run risk analysis on securities from input file.
        Returns a dictionary of analyzed securities or False if analysis fails.
        """
        try:
            # Check if input file exists
            if not os.path.exists(self.input_file):
                self.logger.warning("No securities provided for analysis")
                return False
                
            # Read securities from input file
            with open(self.input_file, 'r') as f:
                securities = json.load(f)
                
            if not securities:
                self.logger.warning("No securities provided for analysis")
                return False
                
            self.logger.info(f"Starting risk analysis for {len(securities)} securities")
            
            # Analyze each security
            results = {}
            for i, security in enumerate(securities, 1):
                ticker = security.get('Ticker', security.get('Symbol'))
                if not ticker:
                    continue
                    
                self.logger.info(f"Processing {ticker} ({i}/{len(securities)})")
                
                # Fetch data and calculate risk score
                yf_data = self._fetch_yf_data(ticker)
                risk_score = self._calculate_risk_score(security)
                
                # Store results
                results[ticker] = {
                    'historical_data': yf_data['history'],
                    'risk_metrics': security,
                    'risk_score': risk_score
                }
                
                self.logger.info(f"Completed analysis for {ticker} with risk score: {risk_score}")
            
            if not results:
                self.logger.warning("No securities analyzed successfully")
                return False
                
            self.logger.info(f"Risk analysis completed. Successfully analyzed {len(results)} out of {len(securities)} securities")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in risk analysis: {str(e)}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    analyzer = ResearchAnalyzer()
    analyzer.run_risk_analysis() 