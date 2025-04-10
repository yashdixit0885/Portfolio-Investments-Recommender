"""
TRADE ANALYZER EXPLANATION
-------------------------

This script analyzes stocks to generate trading recommendations based on technical analysis.
Here's how it works in simple terms:

1. DATA COLLECTION
   - Fetches historical price data for each stock (open, high, low, close prices)
   - Collects trading volume data
   - Uses multiple timeframes (short: 5 days, medium: 20 days, long: 40 days)

2. TECHNICAL INDICATORS
   The script calculates several technical indicators to understand market trends:
   - Trend Indicators: Moving averages, MACD, ADX (shows trend strength)
   - Momentum Indicators: RSI, Stochastic, Williams %R (shows overbought/oversold conditions)
   - Volume Indicators: Money flow, volume trends (shows buying/selling pressure)
   - Volatility Indicators: Bollinger Bands, ATR (shows price volatility)

3. SIGNAL GENERATION
   The script combines these indicators to generate trading signals:
   - BUY: When indicators suggest the stock is likely to go up
   - SELL: When indicators suggest the stock is likely to go down
   - HOLD: When the trend is unclear or neutral

4. CONFIDENCE LEVELS
   Each signal comes with a confidence level:
   - STRONG: Very clear technical signals
   - MODERATE: Good technical signals but not as strong
   - WEAK: Some technical signals but not very clear
   - NEUTRAL: No clear signals in either direction

5. POSITION SIZING
   The script also suggests how much to invest:
   - Calculates position size based on risk metrics
   - Considers volatility and price movements
   - Ensures no single position is too large

6. OUTPUT
   The final recommendations include:
   - Trading signal (BUY/SELL/HOLD)
   - Timeframe (short/medium/long term)
   - Confidence level
   - Position size
   - Detailed justification for the recommendation

The script is designed to help investors make informed decisions by analyzing multiple technical factors
and providing clear, actionable recommendations with proper risk management.
"""

import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from src.utils.common import setup_logging, get_current_time, save_to_json, load_from_json
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import traceback
from src.database import DatabaseManager
from .rate_limiter import RateLimiter
import json

class TradeAnalyzer:
    """
    Swing trading analyzer focusing on medium-term price movements (days to weeks).
    Key features:
    - Swing high/low detection
    - Trend channel analysis
    - Volume confirmation
    - Support/resistance levels
    - Risk management for swing trades
    """

    def __init__(self, db: DatabaseManager):
        """Initialize the TradeAnalyzer with swing trading parameters."""
        self.logger = setup_logging('trade_analyzer')
        self.data_dir = 'data'
        self.output_dir = 'output'
        self.input_file = os.path.join(self.data_dir, 'risk_scored_securities.json')
        self.output_file = os.path.join(self.output_dir, 'Trade_Recommendations_latest.csv')
        self.db = db
        self.rate_limiter = RateLimiter(calls_per_second=2.0, max_retries=3, retry_delay=5.0)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Timeframes for swing trading analysis
        self.timeframes = {
            'short_swing': (5, '1d'),    # 5 days for short-term swings
            'medium_swing': (20, '1d'),  # 20 days for medium-term swings
            'long_swing': (40, '1d')     # 40 days for trend confirmation
        }

        # Signal thresholds
        self.signal_thresholds = {
            'strong': 2.0,
            'moderate': 1.5,
            'weak': 1.0
        }

        # Risk management parameters for swing trading
        self.risk_params = {
            'max_position_size': 0.1,    # Max 10% of portfolio per trade
            'min_risk_reward': 2.0,      # Minimum risk:reward ratio
            'max_drawdown': 0.02,        # Max 2% drawdown per trade
            'min_swing_size': 0.05,      # Minimum 5% price movement for swing
            'max_holding_period': 40      # Maximum 40 days holding period
        }

    def _get_stock_data(self, ticker: str, period: str = '40d', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetches historical stock data with enhanced error handling."""
        self.logger.debug(f"Fetching {period} data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            # Add a buffer to ensure we get enough data
            buffer_period = '60d' if period == '40d' else '30d' if period == '20d' else '10d'
            df = stock.history(period=buffer_period, interval=interval)
            if df.empty:
                self.logger.warning(f"No {period} data found for {ticker}")
                return None
                
            # Take the most recent data according to the requested period
            days_needed = int(period.replace('d', ''))
            if len(df) > days_needed:
                df = df.tail(days_needed)
            
            # Ensure columns are standard and add derived columns
            # Force columns to lowercase after fetching from yfinance
            df.columns = [col.lower() for col in df.columns] 
            df['hl_range'] = df['high'] - df['low']
            df['oc_range'] = df['close'] - df['open']
            df['body_size'] = abs(df['oc_range'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Calculate swing highs and lows
            df = self._calculate_swing_points(df)
            
            self.logger.debug(f"Successfully fetched {len(df)} data points for {ticker}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
            
    def _calculate_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Identifies swing highs and lows in the price data."""
        # Ensure columns are lowercase
        df['swing_high'] = False 
        df['swing_low'] = False
        
        # Find local maxima and minima
        for i in range(window, len(df) - window):
            # Check for swing high
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
                df.loc[df.index[i], 'swing_high'] = True
                
            # Check for swing low
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
                df.loc[df.index[i], 'swing_low'] = True
                
        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculates comprehensive technical indicators for swing trading."""
        if df is None or len(df) < 10:  # Reduced minimum requirement from 20 to 10 days
            self.logger.warning("Insufficient data for swing analysis.")
            return None
            
        self.logger.debug("Calculating technical indicators...")
        try:
            indicators = {}
            
            # Ensure required columns exist and are lowercase
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                 self.logger.error(f"Missing required columns in historical data: {required_cols}")
                 return None

            # Clean and prepare data
            df.dropna(subset=required_cols, inplace=True)
            if df.empty:
                return None
                
            # Adjust window sizes based on available data
            data_len = len(df)
            windows = {
                'short': min(5, data_len - 1),
                'medium': min(10, data_len - 1),
                'long': min(20, data_len - 1)
            }
            
            # --- Calculations using lowercase column names --- 
            
            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(df, windows))
            
            # Momentum Indicators
            indicators.update(self._calculate_momentum_indicators(df, windows))
            
            # Volume Indicators
            indicators.update(self._calculate_volume_indicators(df, windows))
            
            # Volatility Indicators
            indicators.update(self._calculate_volatility_indicators(df, windows))
            
            # Swing Analysis
            indicators.update(self._analyze_swings(df))
            
            # Add current price and recent trend
            indicators['current_price'] = float(df['close'].iloc[-1])
            indicators['recent_trend'] = self._analyze_recent_trend(df)
            
            # Clean up potential NaN/Inf values before returning
            for key, value in indicators.items():
                if isinstance(value, (float, int)) and (pd.isna(value) or np.isinf(value)):
                    indicators[key] = 0.0 # Default to 0 if NaN or Inf
                elif isinstance(value, dict): # Clean nested dicts like recent_trend
                     for sub_key, sub_value in value.items():
                         if isinstance(sub_value, (float, int)) and (pd.isna(sub_value) or np.isinf(sub_value)):
                              value[sub_key] = 0.0

            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return None
            
    def _calculate_trend_indicators(self, df: pd.DataFrame, windows: Dict[str, int]) -> Dict[str, float]:
        """Calculate trend-following indicators."""
        indicators = {}
        
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            window = min(period, windows['long'])
            if window > 0:
                indicators[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=window).iloc[-1]
                indicators[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=window).iloc[-1]
            else:
                indicators[f'sma_{period}'] = 0.0
                indicators[f'ema_{period}'] = 0.0
        
        # MACD
        macd = ta.trend.MACD(df['close'], 
                            window_slow=min(26, windows['medium']),
                            window_fast=min(12, windows['medium']),
                            window_sign=min(9, windows['medium']))
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_hist'] = macd.macd_diff().iloc[-1]
        
        # ADX
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=min(14, windows['medium']))
        indicators['adx'] = adx.adx().iloc[-1]
        indicators['+di'] = adx.adx_pos().iloc[-1]
        indicators['-di'] = adx.adx_neg().iloc[-1]
        
        return indicators

    def _calculate_momentum_indicators(self, df: pd.DataFrame, windows: Dict[str, int]) -> Dict[str, float]:
        """Calculate momentum indicators."""
        indicators = {}
        
        # RSI
        rsi = RSIIndicator(df['close'], window=min(14, windows['medium']))
        indicators['rsi'] = rsi.rsi().iloc[-1]
        
        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'],
                                   window=min(14, windows['medium']),
                                   smooth_window=min(3, windows['short']))
        indicators['stoch_k'] = stoch.stoch().iloc[-1]
        indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
        
        # Williams %R
        willr = WilliamsRIndicator(df['high'], df['low'], df['close'],
                                 lbp=min(14, windows['medium']))
        indicators['willr'] = willr.williams_r().iloc[-1]
        
        return indicators

    def _calculate_volume_indicators(self, df: pd.DataFrame, windows: Dict[str, int]) -> Dict[str, float]:
        """Calculate volume-based indicators."""
        indicators = {}
        
        # OBV
        obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
        indicators['obv'] = obv.on_balance_volume().iloc[-1]
        
        # CMF
        cmf = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'],
                                      window=min(20, windows['short']))
        indicators['cmf'] = cmf.chaikin_money_flow().iloc[-1]
        
        # Volume SMA
        vol_window = min(20, windows['long'])
        if vol_window > 0:
             indicators['volume_sma_20'] = ta.trend.sma_indicator(df['volume'], window=vol_window).iloc[-1]
        else:
             indicators['volume_sma_20'] = 0.0
        
        return indicators

    def _calculate_volatility_indicators(self, df: pd.DataFrame, windows: Dict[str, int]) -> Dict[str, float]:
        """Calculate volatility indicators."""
        indicators = {}
        
        # Bollinger Bands
        bb_window = min(20, windows['short'])
        if bb_window > 0:
            bb = BollingerBands(df['close'], window=bb_window)
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            indicators['bb_width'] = bb.bollinger_wband().iloc[-1]
        else:
            indicators['bb_upper'] = 0.0
            indicators['bb_middle'] = 0.0
            indicators['bb_lower'] = 0.0
            indicators['bb_width'] = 0.0
        
        # ATR
        atr_window = min(14, windows['short'])
        if atr_window > 0:
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=atr_window)
            indicators['atr'] = atr.average_true_range().iloc[-1]
        else:
            indicators['atr'] = 0.0
            
        return indicators

    def _analyze_swings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes swing patterns and trends."""
        analysis = {
            'swing_highs': [],
            'swing_lows': [],
            'trend_channel': {},
            'swing_size': 0.0,
            'swing_direction': 'neutral'
        }
        
        try:
            # Calculate swing points
            window = 5  # Window size for swing detection
            # Use lowercase for new columns
            df['swing_high'] = False 
            df['swing_low'] = False
            
            for i in range(window, len(df) - window):
                # Check for swing high (use lowercase 'high')
                if all(df['high'].iloc[i] > df['high'].iloc[i-window:i]) and \
                   all(df['high'].iloc[i] > df['high'].iloc[i+1:i+window+1]):
                    df.loc[df.index[i], 'swing_high'] = True
                
                # Check for swing low (use lowercase 'low')
                if all(df['low'].iloc[i] < df['low'].iloc[i-window:i]) and \
                   all(df['low'].iloc[i] < df['low'].iloc[i+1:i+window+1]):
                    df.loc[df.index[i], 'swing_low'] = True
            
            # Get recent swing points (use lowercase)
            recent_highs = df[df['swing_high']].tail(3)
            recent_lows = df[df['swing_low']].tail(3)
            
            if not recent_highs.empty:
                analysis['swing_highs'] = recent_highs['high'].tolist()
            
            if not recent_lows.empty:
                analysis['swing_lows'] = recent_lows['low'].tolist()
            
            # Calculate trend channel if we have enough swing points
            if len(analysis['swing_highs']) >= 2 and len(analysis['swing_lows']) >= 2:
                # Upper channel (resistance)
                high_prices = np.array(analysis['swing_highs'])
                high_times = np.arange(len(high_prices))
                high_slope, high_intercept = np.polyfit(high_times, high_prices, 1)
                
                # Lower channel (support)
                low_prices = np.array(analysis['swing_lows'])
                low_times = np.arange(len(low_prices))
                low_slope, low_intercept = np.polyfit(low_times, low_prices, 1)
                
                analysis['trend_channel'] = {
                    'resistance': {'slope': float(high_slope), 'intercept': float(high_intercept)},
                    'support': {'slope': float(low_slope), 'intercept': float(low_intercept)}
                }
                
                # Calculate swing size (channel height)
                channel_height = np.mean(high_prices) - np.mean(low_prices)
                analysis['swing_size'] = float(channel_height)
                
                # Determine swing direction
                if high_slope > 0 and low_slope > 0:
                    analysis['swing_direction'] = 'upward'
                elif high_slope < 0 and low_slope < 0:
                    analysis['swing_direction'] = 'downward'
            else:
                    analysis['swing_direction'] = 'sideways'
            
        except Exception as e:
            self.logger.warning(f"Error in swing analysis: {str(e)}")
            # Return default values if analysis fails
        
        return analysis

    def _analyze_recent_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recent price action and trend characteristics."""
        recent = df.tail(5)
        analysis = {
            'trend_strength': 0.0,
            'trend_direction': 'neutral',
            'price_action': []
        }
        
        # Calculate trend strength
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        
        # Check for higher highs and higher lows (uptrend)
        if all(highs[i] > highs[i-1] for i in range(1, len(highs))) and \
           all(lows[i] > lows[i-1] for i in range(1, len(lows))):
            analysis['trend_strength'] = 1.0
            analysis['trend_direction'] = 'up'
            
        # Check for lower highs and lower lows (downtrend)
        elif all(highs[i] < highs[i-1] for i in range(1, len(highs))) and \
             all(lows[i] < lows[i-1] for i in range(1, len(lows))):
            analysis['trend_strength'] = -1.0
            analysis['trend_direction'] = 'down'
            
        # Analyze candlestick patterns
        for i in range(len(recent)):
            candle = recent.iloc[i]
            # Ensure calculated columns exist and are lowercase
            if 'body_size' in candle and candle['body_size'] > 0:  # Not a doji
                if 'oc_range' in candle and candle['oc_range'] > 0:  # Bullish candle
                    if 'lower_shadow' in candle and candle['lower_shadow'] > 2 * candle['body_size']:
                        analysis['price_action'].append('hammer')
                elif 'oc_range' in candle:  # Bearish candle
                    if 'upper_shadow' in candle and candle['upper_shadow'] > 2 * candle['body_size']:
                        analysis['price_action'].append('shooting_star')
                        
        return analysis

    def _determine_signal_and_timeframe(self, indicators: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """
        Determines swing trading signals based on multiple factors.
        """
        if not indicators:
            return 'HOLD', 'N/A', ['Insufficient data']

        # Initialize scoring system
        scores = {
            'short_swing': 0.0,
            'medium_swing': 0.0,
            'long_swing': 0.0
        }
        justifications = {
            'short_swing': [],
            'medium_swing': [],
            'long_swing': []
        }

        current_price = indicators['current_price']
        swing_analysis = indicators.get('swing_analysis', {})

        # --- Trend Analysis ---
        # ADX for trend strength
        if indicators['adx'] > 25:  # Strong trend
            if indicators['+di'] > indicators['-di']:
                scores['medium_swing'] += 1.0
                justifications['medium_swing'].append('Strong uptrend (ADX > 25)')
            else:
                scores['medium_swing'] -= 1.0
                justifications['medium_swing'].append('Strong downtrend (ADX > 25)')

        # Moving Average Alignment
        if (indicators['ema_10'] > indicators['ema_20'] > indicators['ema_50']):
            scores['medium_swing'] += 1.5
            justifications['medium_swing'].append('Bullish MA alignment')
        elif (indicators['ema_10'] < indicators['ema_20'] < indicators['ema_50']):
            scores['medium_swing'] -= 1.5
            justifications['medium_swing'].append('Bearish MA alignment')

        # --- Swing Analysis ---
        if swing_analysis.get('swing_size', 0) > self.risk_params['min_swing_size']:
            if swing_analysis['swing_direction'] == 'upward':
                scores['short_swing'] += 1.0
                justifications['short_swing'].append('Significant upward swing')
            elif swing_analysis['swing_direction'] == 'downward':
                scores['short_swing'] -= 1.0
                justifications['short_swing'].append('Significant downward swing')

        # --- Momentum Analysis ---
        # RSI with trend context
        rsi = indicators['rsi']
        if rsi < 30 and swing_analysis.get('swing_direction') == 'down':
            scores['short_swing'] += 1.0
            justifications['short_swing'].append('Oversold RSI in downtrend')
        elif rsi > 70 and swing_analysis.get('swing_direction') == 'up':
            scores['short_swing'] -= 1.0
            justifications['short_swing'].append('Overbought RSI in uptrend')

        # --- Volume Analysis ---
        if indicators['cmf'] > 0.1:
            scores['medium_swing'] += 0.5
            justifications['medium_swing'].append('Positive money flow')
        elif indicators['cmf'] < -0.1:
            scores['medium_swing'] -= 0.5
            justifications['medium_swing'].append('Negative money flow')

        # --- Risk Management Integration ---
        # Calculate position size based on ATR
        atr = indicators['atr']
        if atr > 0:
            risk_per_share = atr * 2  # 2 ATR stop loss
            position_size = min(
                self.risk_params['max_position_size'],
                self.risk_params['max_drawdown'] / risk_per_share
            )
            indicators['position_size'] = position_size

        # --- Determine Final Signal ---
        strongest_timeframe = max(scores, key=lambda k: abs(scores[k]))
        max_score = scores[strongest_timeframe]
        
        if abs(max_score) >= self.signal_thresholds['strong']:
            signal = 'BUY' if max_score > 0 else 'SELL'
            confidence = 'STRONG'
        elif abs(max_score) >= self.signal_thresholds['moderate']:
            signal = 'BUY' if max_score > 0 else 'SELL'
            confidence = 'MODERATE'
        elif abs(max_score) >= self.signal_thresholds['weak']:
            signal = 'BUY' if max_score > 0 else 'SELL'
            confidence = 'WEAK'
        else:
            signal = 'HOLD'
            confidence = 'NEUTRAL'

        # Compile final justification
        final_justification = [
            f"{confidence} {signal} signal based on {strongest_timeframe} analysis",
            *justifications[strongest_timeframe]
        ]

        return signal, strongest_timeframe, final_justification

    def generate_trade_signals(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals for the provided list of tickers."""
        self.logger.info(f"Starting trade signal generation for {len(tickers)} tickers...")
        trade_results = {}
        if not tickers:
            self.logger.warning("No tickers provided for trade signal generation.")
            return {}

        try:
            # Fetch necessary data only for the provided tickers
            # Optimize by fetching all securities info once if needed? 
            # Current implementation fetches per ticker.
            
            for ticker in tickers:
                try:
                    # Get security info (might be needed for _generate_signals)
                    security = self.db.get_security(ticker)
                    if not security:
                         self.logger.warning(f"Skipping {ticker}: Could not retrieve security info.")
                         continue
                         
                    # Get historical data
                    historical_data = self.db.get_historical_data(ticker)
                    if historical_data is None or historical_data.empty:
                        self.logger.warning(f"Skipping {ticker}: No historical data found.")
                        continue
                    
                    # Get latest analysis results for this specific ticker
                    investment_analysis = self.db.get_latest_analysis(ticker, 'investment_opportunity')
                    risk_analysis = self.db.get_latest_analysis(ticker, 'risk')
                    
                    # Check if BOTH required analysis results exist for this ticker
                    if not investment_analysis or not risk_analysis:
                        self.logger.warning(f"Skipping {ticker}: Missing required investment or risk analysis results.")
                        continue
                    
                    # Generate trading signals using all gathered data
                    signals = self._generate_signals(security, historical_data, 
                                                  investment_analysis, risk_analysis)
                    
                    # Store trade analysis result in DB
                    success = self.db.insert_analysis_result(
                        ticker=ticker,
                        analysis_type='trade',
                        score=signals.get('signal_score', 0.0), # Use signal_score
                        metrics=signals # Store all generated signal info
                    )
                    if not success:
                         self.logger.error(f"Failed to store trade analysis result for {ticker}.")
                         
                    # Add to results dict regardless of DB storage success?
                    trade_results[ticker] = signals
                    
                except Exception as ticker_error:
                    self.logger.error(f"Error analyzing ticker {ticker} for trade signal: {ticker_error}", exc_info=True)
                    continue # Continue to the next ticker
            
            self.logger.info(f"Trade signal generation completed. Generated signals for {len(trade_results)}/{len(tickers)} provided tickers.")
            return trade_results
            
        except Exception as e:
            self.logger.error(f"Error generating trade signals: {str(e)}", exc_info=True)
            return {}

    def _generate_signals(self, security: Dict[str, Any], 
                         historical_data: pd.DataFrame,
                         investment_analysis: Dict[str, Any],
                         risk_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signals for a security using comprehensive indicators."""
        try:
            # 1. Calculate all necessary technical indicators
            indicators = self._calculate_technical_indicators(historical_data)
            if indicators is None:
                self.logger.warning(f"Skipping signal generation for {security['ticker']}: Failed indicator calculation.")
                return None # Indicate failure to generate signals

            # 2. Determine Signal, Timeframe, and Justification
            signal, timeframe, justification = self._determine_signal_and_timeframe(indicators)
            
            # 3. Calculate final signal score (incorporating investment/risk analysis)
            #    (Keep existing logic for now, might need refinement)
            signal_score = (
                0.3 * investment_analysis.get('score', 0) +  
                0.3 * (1 - risk_analysis.get('risk_score', 0)) +  
                0.2 * (1 if signal == 'BUY' else 0 if signal == 'SELL' else 0.5) + # Use determined signal
                0.2 * (1 if indicators.get('recent_trend', {}).get('trend_direction') == 'up' else 
                       0 if indicators.get('recent_trend', {}).get('trend_direction') == 'down' else 0.5) # Use calculated trend
            )
            signal_score = max(0.0, min(1.0, signal_score)) # Clamp score

            # 4. Construct the final output dictionary
            #    Use the results from _determine_signal_and_timeframe
            output = {
                'signal_score': round(signal_score, 3),
                'signal': signal, # From determine method
                'timeframe': timeframe, # From determine method
                'confidence': justification[0].split(' ')[0] if justification else 'NEUTRAL', # Extract confidence
                'current_price': indicators.get('current_price', 0.0),
                'position_size': indicators.get('position_size', 0.0), # From determine method risk calc
                'justification': json.dumps(justification), # From determine method
                # Optionally include key indicators for context in DB?
                # 'rsi': indicators.get('rsi', 0.0),
                # 'macd': indicators.get('macd', 0.0),
                # 'trend': indicators.get('recent_trend',{}).get('trend_direction', 'neutral'),
                'timestamp': datetime.now().isoformat()
            }
            return output
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {security.get('ticker', 'UNKNOWN')}: {str(e)}", exc_info=True)
            return None # Return None on error


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db = DatabaseManager()
    analyzer = TradeAnalyzer(db)
    analyzer.generate_trade_signals()

# --- Methods Removed/Replaced ---
# Most methods from the previous version were related to swing pattern analysis,
# potential return calculation, risk metrics (Sharpe, VaR), and specific trade parameters
# (stop-loss, take-profit based on ATR/Bollinger Bands) which are not the core focus now.
# The new focus is on generating a directional signal (BUY/SELL/HOLD) and an associated
# timeframe (short/medium/long) based on a scoring of multiple standard technical indicators.