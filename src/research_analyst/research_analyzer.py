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
        Fetches data from yfinance API with caching.
        Returns a dictionary containing 'info' and 'history' data.
        """
        yf_data = {'info': {}, 'history': pd.DataFrame()}
        
        try:
            # Check cache first
            cache_key = f"{ticker}_data"
            if cache_key in self.yf_data_cache:
                self.logger.debug(f"Cache hit for {ticker}")
                return self.yf_data_cache[cache_key]
            
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            
            try:
                info_data = stock.info
                if info_data:
                    yf_data['info'] = info_data
            except Exception as e:
                self.logger.warning(f"Error fetching info data for {ticker}: {str(e)}")
            
            try:
                hist_data = stock.history(period='1y')
                if not hist_data.empty:
                    yf_data['history'] = hist_data
            except Exception as e:
                self.logger.warning(f"Error fetching historical data for {ticker}: {str(e)}")
            
            # Cache the data
            self.yf_data_cache[cache_key] = yf_data
            
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
            'short_interest': 0.0,
            'market_correlation': 0.5,  # New factor
            'trend': 0.5  # New factor
        }

        # Calculate Risk Factors
        # 1. Enhanced Volatility Risk
        vol_metrics = self._calculate_volatility_metrics(yf_history)
        beta = security.get('Beta')
        
        if beta is not None and beta > 0:
            risk_factors['volatility'] = min(1.0, max(0.0, (beta - 0.5) / 2.0))
        elif vol_metrics['std_dev'] is not None:
            # Consider both standard deviation and VaR/CVaR
            std_dev_risk = min(1.0, max(0.0, (vol_metrics['std_dev'] - 0.15) / 0.60))
            var_risk = min(1.0, max(0.0, vol_metrics['var_95'] / 0.05)) if vol_metrics['var_95'] else 0.5
            cvar_risk = min(1.0, max(0.0, vol_metrics['cvar_95'] / 0.08)) if vol_metrics['cvar_95'] else 0.5
            
            risk_factors['volatility'] = (std_dev_risk * 0.4 + var_risk * 0.3 + cvar_risk * 0.3)
            
        # Add volatility trend impact
        if vol_metrics['volatility_trend'] is not None and vol_metrics['volatility_trend'] > 0:
            risk_factors['volatility'] = min(1.0, risk_factors['volatility'] + vol_metrics['volatility_trend'])

        # 2. Market Correlation Risk
        try:
            market_data = self._fetch_yf_data('^GSPC')['history']  # S&P 500
            if not market_data.empty and not yf_history.empty:
                security_returns = yf_history['Close'].pct_change()
                market_returns = market_data['Close'].pct_change()
                # Align dates
                common_dates = security_returns.index.intersection(market_returns.index)
                if len(common_dates) > 30:  # Require at least 30 days of data
                    correlation = security_returns[common_dates].corr(market_returns[common_dates])
                    # Higher correlation means lower diversification benefit
                    risk_factors['market_correlation'] = min(1.0, max(0.0, abs(correlation)))
        except Exception as e:
            self.logger.debug(f"Could not calculate market correlation: {str(e)}")

        # 3. Trend-based Risk
        try:
            if not yf_history.empty:
                prices = yf_history['Close']
                sma_50 = prices.rolling(window=50).mean()
                sma_200 = prices.rolling(window=200).mean()
                
                if len(prices) >= 200:
                    current_price = prices.iloc[-1]
                    price_below_50ma = current_price < sma_50.iloc[-1]
                    price_below_200ma = current_price < sma_200.iloc[-1]
                    death_cross = sma_50.iloc[-1] < sma_200.iloc[-1]
                    
                    trend_risk = 0.5  # Base risk
                    if price_below_50ma and price_below_200ma:
                        trend_risk += 0.3
                    if death_cross:
                        trend_risk += 0.2
                        
                    risk_factors['trend'] = min(1.0, trend_risk)
        except Exception as e:
            self.logger.debug(f"Could not calculate trend risk: {str(e)}")

        # 4. Enhanced Sector Risk
        sector = yf_info.get('sector')
        industry = yf_info.get('industry')
        
        sector_risk_map = {
            'Technology': {
                'Software': 0.85,
                'Hardware': 0.75,
                'Semiconductors': 0.90,
                'Internet Content': 0.85,
                '_default': 0.80
            },
            'Healthcare': {
                'Biotechnology': 0.95,
                'Medical Devices': 0.70,
                'Healthcare Providers': 0.60,
                'Pharmaceuticals': 0.75,
                '_default': 0.75
            },
            'Financial': {
                'Banks': 0.65,
                'Insurance': 0.60,
                'Investment Management': 0.75,
                'FinTech': 0.85,
                '_default': 0.70
            },
            'Consumer Cyclical': {
                'Retail': 0.70,
                'Automotive': 0.75,
                'Entertainment': 0.80,
                '_default': 0.70
            },
            'Energy': {
                'Oil & Gas': 0.85,
                'Renewable Energy': 0.90,
                '_default': 0.80
            },
            '_default': 0.70
        }
        
        if sector:
            sector_risks = sector_risk_map.get(sector, sector_risk_map['_default'])
            if isinstance(sector_risks, dict):
                risk_factors['sector'] = sector_risks.get(industry, sector_risks['_default'])
        else:
                risk_factors['sector'] = sector_risks

        # 5. Debt Risk (Enhanced)
        debt_to_equity = yf_info.get('debtToEquity')
        current_ratio = yf_info.get('currentRatio')
        if debt_to_equity is not None:
            risk_factors['debt'] = min(1.0, max(0.0, (debt_to_equity / 100) / 2.0))
            if current_ratio is not None and current_ratio < 1.0:
                risk_factors['debt'] = min(1.0, risk_factors['debt'] + 0.2)

        # 6. Profitability Risk (Enhanced)
        profit_margin = yf_info.get('profitMargins')
        operating_margin = yf_info.get('operatingMargins')
        if profit_margin is not None:
            risk_factors['profitability'] = min(1.0, max(0.0, (0.20 - profit_margin) / 0.40))
            if operating_margin is not None and operating_margin < 0:
                risk_factors['profitability'] = min(1.0, risk_factors['profitability'] + 0.2)

        # 7. Size Risk (Enhanced)
        market_cap = security.get('Market Cap')
        if market_cap is not None and market_cap > 0:
            log_cap = np.log10(market_cap)
            risk_factors['size'] = min(1.0, max(0.0, (11.3 - log_cap) / (11.3 - 8.7)))
            # Add volume-based size risk
            avg_volume = yf_info.get('averageVolume')
            if avg_volume is not None and avg_volume < 100000:  # Low liquidity threshold
                risk_factors['size'] = min(1.0, risk_factors['size'] + 0.1)

        # 8. RSI Extreme Risk
        rsi_input = security.get('rsi')
        if rsi_input is not None:
            rsi_risk = abs(rsi_input - 0.5) / 0.4
            risk_factors['rsi_extreme'] = min(1.0, max(0.0, rsi_risk))

        # 9. Liquidity Risk
        avg_volume = yf_info.get('averageVolume')
        if avg_volume is not None:
            risk_factors['liquidity'] = min(1.0, max(0.0, (1000000 - avg_volume) / 1000000))

        # 10. Earnings Quality
        earnings_growth = yf_info.get('earningsGrowth')
        revenue_growth = yf_info.get('revenueGrowth')
        if earnings_growth is not None and revenue_growth is not None:
            if earnings_growth < 0 or revenue_growth < 0:
                risk_factors['earnings_quality'] = 0.8
            elif earnings_growth < revenue_growth:
                risk_factors['earnings_quality'] = 0.6

        # 11. Institutional Ownership
        institution_holding = yf_info.get('heldPercentInstitutions')
        if institution_holding is not None:
            if institution_holding < 0.2:
                risk_factors['institutional'] = 0.8
            elif institution_holding < 0.5:
                risk_factors['institutional'] = 0.6

        # 12. Short Interest
        short_ratio = yf_info.get('shortRatio')
        if short_ratio is not None:
            risk_factors['short_interest'] = min(1.0, max(0.0, (short_ratio - 2) / 10))

        # Combine Factors with Updated Weights
        weights = {
            'volatility': 0.20,
            'debt': 0.12,
            'profitability': 0.10,
            'size': 0.08,
            'rsi_extreme': 0.05,
            'liquidity': 0.10,
            'sector': 0.10,
            'earnings_quality': 0.05,
            'institutional': 0.05,
            'short_interest': 0.05,
            'market_correlation': 0.05,
            'trend': 0.05
        }

        total_risk_score_normalized = sum(risk_factors[factor] * weights[factor] for factor in weights)
        final_risk_score = int(round(total_risk_score_normalized * 100))
        
        return min(100, max(0, final_risk_score))  # Ensure score is between 0 and 100

    def run_risk_analysis(self, potential_securities: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run risk analysis on potential securities.
        If potential_securities is not provided, load from input file.
        """
        if potential_securities is None:
            try:
                potential_securities = load_from_json(self.input_file)
            except Exception as e:
                self.logger.error(f"Error loading potential securities: {str(e)}")
                return {}

        if not potential_securities:
            self.logger.warning("No securities provided for analysis")
            return {}

        analysis_results = {}
        total_securities = len(potential_securities)
        processed_count = 0
        
        self.logger.info(f"Starting risk analysis for {total_securities} securities")
        
        for security in potential_securities:
            processed_count += 1
            try:
                symbol = security.get('Ticker') or security.get('Symbol')
                if not symbol:
                    self.logger.warning(f"Missing ticker symbol in security data: {security}")
                    continue

                self.logger.info(f"Processing {symbol} ({processed_count}/{total_securities})")
                
                # Fetch historical data using yfinance
                historical_data = self._fetch_yf_data(symbol)
                if not historical_data or historical_data['history'].empty:
                    self.logger.warning(f"No historical data found for {symbol}")
                    continue
                
                # Calculate risk metrics
                risk_metrics = {
                    'Symbol': symbol,
                    'Beta': historical_data['info'].get('beta', None),
                    'Market Cap': historical_data['info'].get('marketCap', None),
                    'name': historical_data['info'].get('shortName', symbol)
                }
                
                # Skip if essential metrics are missing
                if risk_metrics['Beta'] is None or risk_metrics['Market Cap'] is None:
                    self.logger.warning(f"Missing essential metrics for {symbol}")
                    continue
            
                # Calculate risk score
                risk_score = self._calculate_risk_score(risk_metrics)
                
                # Store results
                analysis_results[symbol] = {
                    'historical_data': historical_data['history'],
                    'risk_metrics': risk_metrics,
                    'risk_score': risk_score
                }
                
                self.logger.info(f"Completed analysis for {symbol} with risk score: {risk_score}")
            except Exception as e:
                self.logger.error(f"Error analyzing security {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Risk analysis completed. Successfully analyzed {len(analysis_results)} out of {total_securities} securities")
        
        # Save results to output file
        try:
            save_to_json(analysis_results, self.output_file)
            return analysis_results
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
            return analysis_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    analyzer = ResearchAnalyzer()
    analyzer.run_risk_analysis() 