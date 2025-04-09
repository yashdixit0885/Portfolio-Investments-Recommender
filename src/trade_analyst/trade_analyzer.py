import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from src.utils.common import setup_logging, get_current_time, save_to_json, load_from_json
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange

class TradeAnalyzer:
    """
    Analyzes risk-scored securities to generate BUY/SELL signals with associated timeframes.
    Focuses on technical analysis to identify potential trade entries based on momentum,
    trend, and oscillator indicators across short, medium, and long terms.
    Outputs results to a single CSV file for user consumption.
    """

    def __init__(self):
        """Initialize the TradeAnalyzer."""
        self.logger = setup_logging('trade_analyzer')
        self.data_dir = 'data'
        self.output_dir = 'output'
        self.input_file = os.path.join(self.data_dir, 'risk_scored_securities.json')
        self.output_file = os.path.join(self.output_dir, 'Trade_Recommendations_latest.csv')

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Timeframes for analysis interpretation (not data fetching period)
        self.timeframes = {
            'short_term': (5, 20),  # Using EMA 5/20, Stochastics
            'medium_term': (20, 50), # Using SMA 20/50, MACD, RSI
            'long_term': (50, 200)   # Using SMA 50/200
        }
        # Minimum absolute score to generate a signal
        self.signal_threshold = 1.5

    def _get_stock_data(self, ticker: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetches historical stock data from yfinance for a given period."""
        self.logger.debug(f"Fetching {period} data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                self.logger.warning(f"No {period} data found for {ticker}")
                return None
            # Ensure columns are standard
            df.columns = [col.capitalize() for col in df.columns]
            self.logger.debug(f"Successfully fetched {len(df)} data points for {ticker}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculates a comprehensive set of technical indicators."""
        if df is None or len(df) < 5:  # Need at least 5 data points for basic indicators
            self.logger.warning("Insufficient data for comprehensive TA.")
            return None
            
        self.logger.debug("Calculating technical indicators...")
        try:
            indicators = {}
            # Clean NaN values before calculating indicators
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            if df.empty:
                self.logger.warning("DataFrame empty after dropping NaN.")
                return None
                
            # Adjust window sizes based on available data
            data_len = len(df)
            window_20 = min(20, data_len - 1)
            window_50 = min(50, data_len - 1)
            window_200 = min(200, data_len - 1)
                
            # Moving Averages
            indicators['sma_20'] = ta.trend.sma_indicator(df['Close'], window=window_20)
            indicators['sma_50'] = ta.trend.sma_indicator(df['Close'], window=window_50)
            indicators['sma_200'] = ta.trend.sma_indicator(df['Close'], window=window_200)
            indicators['ema_5'] = ta.trend.ema_indicator(df['Close'], window=5)
            indicators['ema_10'] = ta.trend.ema_indicator(df['Close'], window=min(10, data_len - 1))
            indicators['ema_20'] = ta.trend.ema_indicator(df['Close'], window=window_20)
            indicators['ema_50'] = ta.trend.ema_indicator(df['Close'], window=window_50)

            # MACD (12, 26, 9)
            macd_ind = ta.trend.MACD(df['Close'], 
                                   window_slow=min(26, data_len - 1),
                                   window_fast=min(12, data_len - 1),
                                   window_sign=min(9, data_len - 1))
            indicators['macd'] = macd_ind.macd()
            indicators['macd_signal'] = macd_ind.macd_signal()
            indicators['macd_hist'] = macd_ind.macd_diff() # Histogram

            # RSI (14)
            indicators['rsi'] = ta.momentum.RSIIndicator(df['Close'], 
                                                       window=min(14, data_len - 1)).rsi()

            # Stochastic Oscillator (14, 3)
            stoch_ind = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'],
                                                        window=min(14, data_len - 1),
                                                        smooth_window=min(3, data_len - 1))
            indicators['stoch_k'] = stoch_ind.stoch()
            indicators['stoch_d'] = stoch_ind.stoch_signal()

            # Bollinger Bands (20)
            bb_ind = ta.volatility.BollingerBands(df['Close'], window=window_20)
            indicators['bb_hband'] = bb_ind.bollinger_hband()
            indicators['bb_mavg'] = bb_ind.bollinger_mavg()
            indicators['bb_lband'] = bb_ind.bollinger_lband()
            indicators['bb_width'] = bb_ind.bollinger_wband() # Bandwidth
            
            # Return only the latest value for each indicator
            latest_indicators = {key: float(series.iloc[-1]) if not series.empty and not pd.isna(series.iloc[-1]) else 0.0
                               for key, series in indicators.items()}
            
            # Add current price for context
            latest_indicators['current_price'] = float(df['Close'].iloc[-1])

            self.logger.debug("Finished calculating technical indicators.")
            return latest_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}", exc_info=True)
            return None
            
    def _determine_signal_and_timeframe(self, indicators: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """
        Evaluates indicators to determine the strongest signal (BUY/SELL/HOLD) 
        and associated timeframe (short_term, medium_term, long_term).
        Returns: (Signal, Timeframe, Justification List)
        """
        if not indicators or pd.isna(indicators.get('current_price')):
            return 'HOLD', 'N/A', ['Insufficient indicator data']

        scores = {'short_term': 0.0, 'medium_term': 0.0, 'long_term': 0.0}
        justifications = {'short_term': [], 'medium_term': [], 'long_term': []}

        price = indicators['current_price']

        # --- Short-Term Factors (EMA 5/10/20, Stochastics, RSI extremes, BB) ---
        if not pd.isna(indicators.get('ema_5')) and not pd.isna(indicators.get('ema_10')):
            if indicators['ema_5'] > indicators['ema_10']:
                scores['short_term'] += 1.0
                justifications['short_term'].append('EMA(5) > EMA(10)')
            else:
                scores['short_term'] -= 1.0
                justifications['short_term'].append('EMA(5) < EMA(10)')
        
        stoch_k = indicators.get('stoch_k')
        stoch_d = indicators.get('stoch_d')
        if not pd.isna(stoch_k) and not pd.isna(stoch_d):
            if stoch_k < 20 and stoch_d < 20 and stoch_k > stoch_d: # Oversold bullish cross
                scores['short_term'] += 1.0
                justifications['short_term'].append('Stoch Oversold Bullish Cross')
            elif stoch_k > 80 and stoch_d > 80 and stoch_k < stoch_d: # Overbought bearish cross
                scores['short_term'] -= 1.0
                justifications['short_term'].append('Stoch Overbought Bearish Cross')
            elif stoch_k < 30:
                 scores['short_term'] += 0.5
                 justifications['short_term'].append('Stoch < 30')
            elif stoch_k > 70:
                 scores['short_term'] -= 0.5
                 justifications['short_term'].append('Stoch > 70')

        rsi = indicators.get('rsi')
        if not pd.isna(rsi):
            if rsi < 30:
                scores['short_term'] += 0.5
                justifications['short_term'].append('RSI < 30')
            elif rsi > 70:
                scores['short_term'] -= 0.5
                justifications['short_term'].append('RSI > 70')
                
        bb_lband = indicators.get('bb_lband')
        bb_hband = indicators.get('bb_hband')
        if not pd.isna(bb_lband) and not pd.isna(bb_hband):
             if price < bb_lband:
                 scores['short_term'] += 0.5
                 justifications['short_term'].append('Price < Lower BB')
             elif price > bb_hband:
                 scores['short_term'] -= 0.5
                 justifications['short_term'].append('Price > Upper BB')

        # --- Medium-Term Factors (SMA 20/50, MACD, RSI trend) ---
        if not pd.isna(indicators.get('sma_20')) and not pd.isna(indicators.get('sma_50')):
            if indicators['sma_20'] > indicators['sma_50']:
                scores['medium_term'] += 1.0
                justifications['medium_term'].append('SMA(20) > SMA(50)')
            else:
                scores['medium_term'] -= 1.0
                justifications['medium_term'].append('SMA(20) < SMA(50)')

        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        macd_hist = indicators.get('macd_hist')
        if not pd.isna(macd) and not pd.isna(macd_signal) and not pd.isna(macd_hist):
            if macd > macd_signal: # Bullish crossover/stance
                scores['medium_term'] += 1.0
                justifications['medium_term'].append('MACD Line > Signal Line')
                if macd_hist > 0: # Optional: check if histogram confirms
                     scores['medium_term'] += 0.5
                     justifications['medium_term'].append('MACD Hist > 0')
            else: # Bearish crossover/stance
                scores['medium_term'] -= 1.0
                justifications['medium_term'].append('MACD Line < Signal Line')
                if macd_hist < 0:
                     scores['medium_term'] -= 0.5
                     justifications['medium_term'].append('MACD Hist < 0')
                     
        if not pd.isna(rsi):
            if rsi > 55: # Trending bullish
                scores['medium_term'] += 0.5
                justifications['medium_term'].append('RSI > 55')
            elif rsi < 45: # Trending bearish
                scores['medium_term'] -= 0.5
                justifications['medium_term'].append('RSI < 45')

        # --- Long-Term Factors (SMA 50/200) ---
        sma_50 = indicators.get('sma_50')
        sma_200 = indicators.get('sma_200')
        if not pd.isna(sma_50) and not pd.isna(sma_200):
            if sma_50 > sma_200: # Golden cross territory
                scores['long_term'] += 1.5 # Stronger signal
                justifications['long_term'].append('SMA(50) > SMA(200)')
            else: # Death cross territory
                scores['long_term'] -= 1.5
                justifications['long_term'].append('SMA(50) < SMA(200)')
            
            if price > sma_200: # Price above long-term average
                scores['long_term'] += 1.0
                justifications['long_term'].append('Price > SMA(200)')
            else:
                 scores['long_term'] -= 1.0
                 justifications['long_term'].append('Price < SMA(200)')

        # --- Determine Dominant Signal --- 
        strongest_timeframe = 'N/A'
        max_score = 0.0
        final_signal = 'HOLD'
        final_justification = ['Neutral signals or below threshold']

        for timeframe, score in scores.items():
            if abs(score) > abs(max_score):
                max_score = score
                strongest_timeframe = timeframe
        
        if abs(max_score) >= self.signal_threshold:
            final_signal = 'BUY' if max_score > 0 else 'SELL'
            final_justification = justifications[strongest_timeframe]
            # Add confirmation from other timeframes if they align
            for timeframe, score in scores.items():
                 if timeframe != strongest_timeframe: 
                     if (max_score > 0 and score > 0) or (max_score < 0 and score < 0):
                         final_justification.append(f"Aligns with {timeframe}")
        
        self.logger.debug(f"Scores - Short: {scores['short_term']:.1f}, Med: {scores['medium_term']:.1f}, Long: {scores['long_term']:.1f}")
        self.logger.debug(f"Result -> Signal: {final_signal}, Timeframe: {strongest_timeframe}, Score: {max_score:.1f}")
        
        return final_signal, strongest_timeframe, final_justification

    def generate_trade_signals(self) -> bool:
        """Loads risk-scored data, analyzes technicals, determines signals, and saves results."""
        self.logger.info("Starting Trade Analyzer: Generating trade signals.")

        # Load securities from ResearchAnalyzer
        if not os.path.exists(self.input_file):
            self.logger.error(f"Input file not found: {self.input_file}")
            return False

        input_data = load_from_json(self.input_file)
        if not input_data or "risk_scored_securities" not in input_data:
            self.logger.warning(f"No valid securities found in {self.input_file}. Nothing to analyze.")
            return False
            
        securities = input_data["risk_scored_securities"]
        self.logger.info(f"Loaded {len(securities)} risk-scored securities for trade signal analysis.")

        trade_recommendations = []
        for security in securities:
            ticker = security.get('Symbol')
            risk_score = security.get('risk_score', 'N/A')
            if not ticker:
                self.logger.warning("Skipping security with missing Symbol.")
                continue

            self.logger.info(f"Analyzing {ticker} (Risk: {risk_score})...")
            # Fetch 1 year of data for comprehensive TA
            stock_data = self._get_stock_data(ticker, period='1y')
            if stock_data is None:
                self.logger.warning(f"Skipping {ticker} due to data fetching error.")
                continue
            
            # Calculate indicators
            indicators = self._calculate_technical_indicators(stock_data)
            if indicators is None:
                 self.logger.warning(f"Skipping {ticker} due to indicator calculation error.")
                 continue
                 
            # Determine signal
            signal, timeframe, justification = self._determine_signal_and_timeframe(indicators)

            # Add to recommendations list
            recommendation = {
                'Ticker': ticker,
                'Signal': signal,
                'Timeframe': timeframe,
                'Risk Score': risk_score,
                'Current Price': indicators.get('current_price', 'N/A'),
                'Justification': "; ".join(justification), # Combine justifications
                'Analysis Timestamp': get_current_time().isoformat()
            }
            trade_recommendations.append(recommendation)

        # Save results to CSV
        if not trade_recommendations:
            self.logger.warning("No trade recommendations generated.")
            return False

        try:
            recommendations_df = pd.DataFrame(trade_recommendations)
            recommendations_df.to_csv(self.output_file, index=False)
            self.logger.info(f"Saved {len(trade_recommendations)} trade recommendations to {self.output_file}")
            self.logger.info("Trade Analyzer finished successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving trade recommendations to CSV: {str(e)}")
            return False


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    analyzer = TradeAnalyzer()
    analyzer.generate_trade_signals()

# --- Methods Removed/Replaced ---
# Most methods from the previous version were related to swing pattern analysis,
# potential return calculation, risk metrics (Sharpe, VaR), and specific trade parameters
# (stop-loss, take-profit based on ATR/Bollinger Bands) which are not the core focus now.
# The new focus is on generating a directional signal (BUY/SELL/HOLD) and an associated
# timeframe (short/medium/long) based on a scoring of multiple standard technical indicators.