"""
Research Analyzer: Evaluating Investment Risk and Opportunities

This module analyzes securities for potential investment opportunities and risks.
It combines fundamental analysis, technical indicators, and market sentiment to
provide a comprehensive view of each security.
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import yfinance as yf
from src.database import DatabaseManager
from .rate_limiter import RateLimiter

class ResearchAnalyzer:
    """Analyzes securities for investment opportunities and risks."""
    
    def __init__(self, db: DatabaseManager):
        """Initialize the ResearchAnalyzer."""
        self.logger = logging.getLogger('research_analyzer')
        self.db = db
        self.rate_limiter = RateLimiter(calls_per_second=2.0, max_retries=3, retry_delay=5.0)
        
    def run_risk_analysis(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run risk analysis on the provided list of tickers."""
        self.logger.info(f"Starting risk analysis cycle for {len(tickers)} tickers...")
        risk_results = {}
        if not tickers:
            self.logger.warning("No tickers provided for risk analysis.")
            return {}

        try:
            # Get security details from DB only for the provided tickers
            # This could be optimized with a single query if db_manager supports it
            securities_info = {s['ticker']: s for s in self.db.get_all_securities() if s['ticker'] in tickers}
            if not securities_info:
                 self.logger.warning("Could not retrieve security info from DB for provided tickers.")
                 # Or handle partial failure?
                 return {}
                
            for ticker in tickers:
                security = securities_info.get(ticker)
                if not security:
                    self.logger.warning(f"Missing security info for {ticker}, skipping risk analysis.")
                    continue
                
                try:
                    self.logger.debug(f"Analyzing risk for {ticker}...")
                    historical_data = self.db.get_historical_data(ticker)
                    
                    if historical_data is None or historical_data.empty:
                        self.logger.warning(f"Skipping {ticker}: No historical data found for risk analysis.")
                        continue
                        
                    # Calculate risk metrics - Pass security dict and historical data
                    risk_metrics = self._calculate_risk_metrics(security, historical_data)
                    if risk_metrics is None:
                         self.logger.warning(f"Skipping {ticker}: Failed to calculate risk metrics.")
                         continue 
            
                    # Store results
                    success = self.db.insert_analysis_result(
                        ticker=ticker,
                        analysis_type='risk',
                        score=risk_metrics.get('risk_score', 0.0), 
                        metrics=risk_metrics
                    )
                    if not success:
                         self.logger.error(f"Failed to store risk analysis result for {ticker}.")
                    else:
                         # Only add to results if successfully stored?
                         risk_results[ticker] = risk_metrics
                         self.logger.info(f"Completed risk analysis for {ticker}. Score: {risk_metrics.get('risk_score'):.2f}")

                except Exception as ticker_error:
                     self.logger.error(f"Error running risk analysis for {ticker}: {ticker_error}", exc_info=True)
                     continue # Move to next ticker

            self.logger.info(f"Risk analysis cycle completed. Analyzed {len(risk_results)}/{len(tickers)} provided tickers.")
            return risk_results
        except Exception as e:
            self.logger.error(f"Error in risk analysis cycle: {str(e)}", exc_info=True)
            return {}
            
    def _calculate_risk_metrics(self, security: Dict[str, Any], 
                              historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for a security."""
        try:
            # Ensure columns are lowercase
            if 'close' not in historical_data.columns or 'volume' not in historical_data.columns:
                 self.logger.error("Historical data missing required 'close' or 'volume' columns.")
                 # Return default risk metrics if columns missing
                 return {
                    'risk_score': 1.0, 'volatility': 0.0, 'max_drawdown': 0.0, 'beta': 0.0,
                    'avg_volume': 0.0, 'volume_volatility': 0.0, 'institutional_ownership': 0.0,
                    'timestamp': datetime.now().isoformat()
                 }
            
            # Calculate volatility
            returns = historical_data['close'].pct_change()
            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
            
            # Calculate drawdown
            rolling_max = historical_data['close'].expanding().max()
            drawdown = (historical_data['close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate beta
            beta = security.get('beta', 0)
            
            # Calculate liquidity metrics
            avg_volume = historical_data['volume'].mean()
            # Handle division by zero if avg_volume is zero
            volume_volatility = 0.0 if avg_volume == 0 else historical_data['volume'].std() / avg_volume
            
            # Fill NaN values that might arise from calculations (e.g., std dev with 1 data point)
            volatility = 0.0 if pd.isna(volatility) else volatility
            max_drawdown = 0.0 if pd.isna(max_drawdown) else max_drawdown
            volume_volatility = 0.0 if pd.isna(volume_volatility) else volume_volatility
            beta = 0.0 if pd.isna(beta) else beta
            inst_own_pct = security.get('inst_own_pct', 0)
            inst_own_pct = 0.0 if pd.isna(inst_own_pct) else inst_own_pct

            # Calculate risk score (0-1, higher is riskier)
            risk_score = (
                0.3 * min(volatility / 0.5, 1.0) +  # Volatility component
                0.2 * min(abs(max_drawdown) / 0.5, 1.0) +  # Drawdown component
                0.2 * min(abs(beta) / 2.0, 1.0) +  # Beta component
                0.2 * min(volume_volatility / 2.0, 1.0) +  # Volume volatility component
                0.1 * (1 - min(inst_own_pct, 1.0))  # Institutional ownership component
            )
            
            return {
                'risk_score': risk_score,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'beta': beta,
                'avg_volume': avg_volume,
                'volume_volatility': volume_volatility,
                'institutional_ownership': inst_own_pct,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'risk_score': 1.0,  # Maximum risk if calculation fails
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'beta': 0.0,
                'avg_volume': 0.0,
                'volume_volatility': 0.0,
                'institutional_ownership': 0.0,
                'timestamp': datetime.now().isoformat()
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db = DatabaseManager()
    analyzer = ResearchAnalyzer(db)
    analyzer.run_risk_analysis() 