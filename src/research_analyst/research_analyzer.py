"""
Research Analyzer: Risk Scoring Logic Overview (Business)

This script analyzes securities that show high potential for price movement 
and assigns a Risk Score from 0 (lowest risk) to 100 (highest risk). 
The goal is to help prioritize which securities might be safer or riskier 
for trading.
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
        self.data_dir = 'data'
        # Input file from InvestmentAnalyzer
        self.input_file = os.path.join(self.data_dir, 'high_potential_securities.json')
        # Output file for TradeAnalyst
        self.output_file = os.path.join(self.data_dir, 'risk_scored_securities.json')

        # Rate limiter for external API calls
        self.rate_limiter = RateLimiter(calls_per_second=2.0, max_retries=3, retry_delay=5.0)
        
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
                # Fetch 1 year for volatility/beta calculations
                hist_data = yf_stock.history(period='1y')
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
        """Calculates annualized standard deviation from price history."""
        volatility = {'std_dev': None}
        if not history.empty and 'Close' in history.columns and len(history) > 1:
            try:
                daily_returns = history['Close'].pct_change().dropna()
                if not daily_returns.empty:
                    # Annualized standard deviation
                    std_dev = daily_returns.std() * np.sqrt(252)
                    volatility['std_dev'] = round(std_dev, 4)
            except Exception as e:
                self.logger.warning(f"Could not calculate volatility: {str(e)}")
        return volatility

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
            'rsi_extreme': 0.0  # Default to low risk if not extreme
        }

        # Calculate Risk Factors
        beta = security.get('Beta')
        std_dev = None
        if not yf_history.empty:
            vol_metrics = self._calculate_volatility_metrics(yf_history)
            std_dev = vol_metrics.get('std_dev')

        if beta is not None and beta > 0:
            risk_factors['volatility'] = min(1.0, max(0.0, (beta - 0.5) / 2.0))
            self.logger.debug(f"{ticker}: Using Beta {beta:.2f} -> Volatility Risk {risk_factors['volatility']:.2f}")
        elif std_dev is not None and std_dev > 0:
            risk_factors['volatility'] = min(1.0, max(0.0, (std_dev - 0.15) / 0.60))
            self.logger.debug(f"{ticker}: Using Std Dev {std_dev:.2f} -> Volatility Risk {risk_factors['volatility']:.2f}")

        # Debt Risk
        debt_to_equity = yf_info.get('debtToEquity')
        if debt_to_equity is not None:
            risk_factors['debt'] = min(1.0, max(0.0, (debt_to_equity / 100) / 2.0))
            self.logger.debug(f"{ticker}: D/E {debt_to_equity:.1f} -> Debt Risk {risk_factors['debt']:.2f}")

        # Profitability Risk
        profit_margin = yf_info.get('profitMargins')
        if profit_margin is not None:
            risk_factors['profitability'] = min(1.0, max(0.0, (0.20 - profit_margin) / 0.40))
            self.logger.debug(f"{ticker}: Margin {profit_margin:.2%} -> Profit Risk {risk_factors['profitability']:.2f}")

        # Size Risk
        market_cap = security.get('Market Cap')
        if market_cap is not None and market_cap > 0:
            log_cap = np.log10(market_cap)
            risk_factors['size'] = min(1.0, max(0.0, (11.3 - log_cap) / (11.3 - 8.7)))
            self.logger.debug(f"{ticker}: Cap {market_cap:,.0f} (log={log_cap:.1f}) -> Size Risk {risk_factors['size']:.2f}")

        # RSI Extreme Risk
        rsi_input = security.get('rsi')
        if rsi_input is not None:
            rsi_risk = abs(rsi_input - 0.5) / 0.4
            risk_factors['rsi_extreme'] = min(1.0, max(0.0, rsi_risk))
            self.logger.debug(f"{ticker}: RSI {rsi_input:.2f} -> RSI Extreme Risk {risk_factors['rsi_extreme']:.2f}")

        # Combine Factors with Weights
        weights = {
            'volatility': 0.30,
            'debt': 0.25,
            'profitability': 0.20,
            'size': 0.15,
            'rsi_extreme': 0.10
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