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
from typing import Dict, List, Any, Optional

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
        self.yf_data_cache = {}

    def _fetch_yf_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches necessary data (info and history) from yfinance for a ticker, 
        using rate limiting and caching.
        """
        if ticker in self.yf_data_cache:
            return self.yf_data_cache[ticker]
            
        self.logger.debug(f"Fetching yfinance data for {ticker}")
        yf_data = {'info': {}, 'history': pd.DataFrame()}
        try:
            yf_stock = yf.Ticker(ticker)

            # Define callables for rate limiter
            def get_info():
                # Basic validation within the call
                info_data = yf_stock.info
                # Check for minimal viability - requires a market price
                if not info_data or not info_data.get('regularMarketPrice'):
                    self.logger.warning(f"No valid market data found for {ticker} via yfinance info.")
                    return {}
                return info_data

            def get_history():
                # Fetch 2 years for better volatility/beta calculations
                hist_data = yf_stock.history(period='2y')
                if hist_data.empty:
                    self.logger.warning(f"No valid history found for {ticker} via yfinance.")
                    return pd.DataFrame()
                return hist_data

            # Use rate limiter
            yf_data['info'] = self.rate_limiter.call_with_retry(get_info)
            # Only fetch history if info was minimally successful (implies valid ticker)
            if yf_data['info']:
                yf_data['history'] = self.rate_limiter.call_with_retry(get_history)
            
        except Exception as e:
            # Handle cases where yfinance might raise exceptions for invalid tickers etc.
            self.logger.error(f"Error fetching yfinance data for {ticker}: {str(e)}")
            # Return empty structures on error
            yf_data = {'info': {}, 'history': pd.DataFrame()}
            
        self.yf_data_cache[ticker] = yf_data
        return yf_data

    def _calculate_volatility_metrics(self, history: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculates comprehensive volatility metrics from price history."""
        metrics = {
            'std_dev': None,
            'max_drawdown': None,
            'upside_vol': None,
            'downside_vol': None
        }
        
        if not history.empty and 'Close' in history.columns and len(history) > 1:
            try:
                daily_returns = history['Close'].pct_change().dropna()
                if not daily_returns.empty:
                    # Annualized standard deviation
                    metrics['std_dev'] = daily_returns.std() * np.sqrt(252)
                    
                    # Maximum drawdown
                    cummax = history['Close'].cummax()
                    drawdown = (history['Close'] - cummax) / cummax
                    metrics['max_drawdown'] = abs(drawdown.min())
                    
                    # Upside and downside volatility
                    positive_returns = daily_returns[daily_returns > 0]
                    negative_returns = daily_returns[daily_returns < 0]
                    
                    metrics['upside_vol'] = positive_returns.std() * np.sqrt(252) if not positive_returns.empty else 0
                    metrics['downside_vol'] = abs(negative_returns.std() * np.sqrt(252)) if not negative_returns.empty else 0
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
        yf_history = yf_data.get('history', pd.DataFrame())

        # Risk Factors Initialization (0 = lowest risk, 1 = highest risk)
        risk_factors = {
            'volatility': 0.5,  # Default to medium risk
            'debt': 0.5,
            'profitability': 0.5,
            'size': 0.5,
            'rsi_extreme': 0.0,
            'liquidity': 0.5,
            'sector': 0.5,
            'earnings_quality': 0.5,
            'institutional': 0.5,
            'short_interest': 0.0
        }

        # Calculate Risk Factors
        # 1. Volatility Risk (Enhanced)
        beta = security.get('Beta')
        vol_metrics = self._calculate_volatility_metrics(yf_history)
        
        if beta is not None and beta > 0:
            risk_factors['volatility'] = min(1.0, max(0.0, (beta - 0.5) / 2.0))
        elif vol_metrics['std_dev'] is not None:
            risk_factors['volatility'] = min(1.0, max(0.0, (vol_metrics['std_dev'] - 0.15) / 0.60))
            
        # Add max drawdown risk
        if vol_metrics['max_drawdown'] is not None:
            drawdown_risk = min(1.0, max(0.0, vol_metrics['max_drawdown'] / 0.5))
            risk_factors['volatility'] = max(risk_factors['volatility'], drawdown_risk)

        # 2. Debt Risk (Enhanced)
        debt_to_equity = yf_info.get('debtToEquity')
        current_ratio = yf_info.get('currentRatio')
        if debt_to_equity is not None:
            risk_factors['debt'] = min(1.0, max(0.0, (debt_to_equity / 100) / 2.0))
            if current_ratio is not None and current_ratio < 1.0:
                risk_factors['debt'] = min(1.0, risk_factors['debt'] + 0.2)

        # 3. Profitability Risk (Enhanced)
        profit_margin = yf_info.get('profitMargins')
        operating_margin = yf_info.get('operatingMargins')
        if profit_margin is not None:
            risk_factors['profitability'] = min(1.0, max(0.0, (0.20 - profit_margin) / 0.40))
            if operating_margin is not None and operating_margin < 0:
                risk_factors['profitability'] = min(1.0, risk_factors['profitability'] + 0.2)

        # 4. Size Risk (Enhanced)
        market_cap = security.get('Market Cap')
        if market_cap is not None and market_cap > 0:
            log_cap = np.log10(market_cap)
            risk_factors['size'] = min(1.0, max(0.0, (11.3 - log_cap) / (11.3 - 8.7)))
            # Add volume-based size risk
            avg_volume = yf_info.get('averageVolume')
            if avg_volume is not None and avg_volume < 100000:  # Low liquidity threshold
                risk_factors['size'] = min(1.0, risk_factors['size'] + 0.1)

        # 5. RSI Extreme Risk
        rsi_input = security.get('rsi')
        if rsi_input is not None:
            rsi_risk = abs(rsi_input - 0.5) / 0.4
            risk_factors['rsi_extreme'] = min(1.0, max(0.0, rsi_risk))

        # 6. Liquidity Risk
        avg_volume = yf_info.get('averageVolume')
        if avg_volume is not None:
            risk_factors['liquidity'] = min(1.0, max(0.0, (1000000 - avg_volume) / 1000000))

        # 7. Sector Risk
        sector = yf_info.get('sector')
        if sector:
            # Higher risk for more volatile sectors
            high_risk_sectors = {'Technology', 'Healthcare', 'Biotechnology'}
            medium_risk_sectors = {'Consumer Cyclical', 'Communication Services', 'Financial Services'}
            if sector in high_risk_sectors:
                risk_factors['sector'] = 0.8
            elif sector in medium_risk_sectors:
                risk_factors['sector'] = 0.6

        # 8. Earnings Quality
        earnings_growth = yf_info.get('earningsGrowth')
        revenue_growth = yf_info.get('revenueGrowth')
        if earnings_growth is not None and revenue_growth is not None:
            if earnings_growth < 0 or revenue_growth < 0:
                risk_factors['earnings_quality'] = 0.8
            elif earnings_growth < revenue_growth:
                risk_factors['earnings_quality'] = 0.6

        # 9. Institutional Ownership
        institution_holding = yf_info.get('heldPercentInstitutions')
        if institution_holding is not None:
            if institution_holding < 0.2:
                risk_factors['institutional'] = 0.8
            elif institution_holding < 0.5:
                risk_factors['institutional'] = 0.6

        # 10. Short Interest
        short_ratio = yf_info.get('shortRatio')
        if short_ratio is not None:
            risk_factors['short_interest'] = min(1.0, max(0.0, (short_ratio - 2) / 10))

        # Combine Factors with Weights
        weights = {
            'volatility': 0.20,
            'debt': 0.15,
            'profitability': 0.12,
            'size': 0.10,
            'rsi_extreme': 0.08,
            'liquidity': 0.10,
            'sector': 0.08,
            'earnings_quality': 0.07,
            'institutional': 0.05,
            'short_interest': 0.05
        }

        total_risk_score_normalized = sum(risk_factors[factor] * weights[factor] for factor in weights)
        final_risk_score = int(round(total_risk_score_normalized * 100))

        self.logger.info(f"Calculated risk score for {ticker}: {final_risk_score}")
        return final_risk_score

    def run_risk_analysis(self) -> bool:
        """Main entry point to load high-potential securities, calculate risk, rank, and save."""
        self.logger.info("Starting Research Analyzer: Calculating risk scores.")
        
        if not os.path.exists(self.input_file):
            self.logger.error(f"Input file not found: {self.input_file}")
            return False
                
        potential_securities = load_from_json(self.input_file)
        if not potential_securities:
            self.logger.warning(f"No securities loaded from {self.input_file}. Nothing to analyze.")
            return False
                
        self.logger.info(f"Loaded {len(potential_securities)} high-potential securities for risk analysis.")

        risk_scored_results = []
        for security in potential_securities:
            security_data = security.copy()
            try:
                risk_score = self._calculate_risk_score(security_data)
                security_data['risk_score'] = risk_score
                risk_scored_results.append(security_data)
            except Exception as e:
                self.logger.error(f"Failed to calculate risk score for {security_data.get('Symbol', 'Unknown')}: {str(e)}", exc_info=True)

        risk_scored_results.sort(key=lambda x: x.get('risk_score', 999))

        if not risk_scored_results:
            self.logger.warning("No securities were successfully risk-scored.")
            return False

        try:
            output_data = {
                "analysis_timestamp": get_current_time().isoformat(),
                "risk_scored_securities": risk_scored_results
            }
            if save_to_json(output_data, self.output_file):
                self.logger.info(f"Saved {len(risk_scored_results)} risk-scored securities to {self.output_file}")
                self.logger.info("Research Analyzer finished successfully.")
                return True
            else:
                self.logger.error(f"Failed to save risk-scored securities to {self.output_file}")
                return False
        except Exception as e:
            self.logger.error(f"Error saving risk-scored securities: {str(e)}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    analyzer = ResearchAnalyzer()
    analyzer.run_risk_analysis() 