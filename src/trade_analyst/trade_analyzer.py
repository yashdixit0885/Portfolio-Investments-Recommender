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

    def __init__(self):
        """Initialize the TradeAnalyzer with swing trading parameters."""
        self.logger = setup_logging('trade_analyzer')
        self.data_dir = 'data'
        self.output_dir = 'output'
        self.input_file = os.path.join(self.data_dir, 'risk_scored_securities.json')
        self.output_file = os.path.join(self.output_dir, 'Trade_Recommendations_latest.csv')

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
            df.columns = [col.capitalize() for col in df.columns]
            df['HL_Range'] = df['High'] - df['Low']
            df['OC_Range'] = df['Close'] - df['Open']
            df['Body_Size'] = abs(df['OC_Range'])
            df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            # Calculate swing highs and lows
            df = self._calculate_swing_points(df)
            
            self.logger.debug(f"Successfully fetched {len(df)} data points for {ticker}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
            
    def _calculate_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Identifies swing highs and lows in the price data."""
        df['Swing_High'] = False
        df['Swing_Low'] = False
        
        # Find local maxima and minima
        for i in range(window, len(df) - window):
            # Check for swing high
            if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, window+1)):
                df.loc[df.index[i], 'Swing_High'] = True
                
            # Check for swing low
            if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, window+1)):
                df.loc[df.index[i], 'Swing_Low'] = True
                
        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculates comprehensive technical indicators for swing trading."""
        if df is None or len(df) < 10:  # Reduced minimum requirement from 20 to 10 days
            self.logger.warning("Insufficient data for swing analysis.")
            return None
            
        self.logger.debug("Calculating technical indicators...")
        try:
            indicators = {}
            
            # Clean and prepare data
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            if df.empty:
                return None
                
            # Adjust window sizes based on available data
            data_len = len(df)
            windows = {
                'short': min(5, data_len - 1),
                'medium': min(10, data_len - 1),
                'long': min(20, data_len - 1)
            }
            
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
            indicators['current_price'] = float(df['Close'].iloc[-1])
            indicators['recent_trend'] = self._analyze_recent_trend(df)
            
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
            indicators[f'sma_{period}'] = ta.trend.sma_indicator(df['Close'], window=window).iloc[-1]
            indicators[f'ema_{period}'] = ta.trend.ema_indicator(df['Close'], window=window).iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(df['Close'], 
                            window_slow=min(26, windows['medium']),
                            window_fast=min(12, windows['medium']),
                            window_sign=min(9, windows['medium']))
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_hist'] = macd.macd_diff().iloc[-1]
        
        # ADX
        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=min(14, windows['medium']))
        indicators['adx'] = adx.adx().iloc[-1]
        indicators['+di'] = adx.adx_pos().iloc[-1]
        indicators['-di'] = adx.adx_neg().iloc[-1]
        
        return indicators

    def _calculate_momentum_indicators(self, df: pd.DataFrame, windows: Dict[str, int]) -> Dict[str, float]:
        """Calculate momentum indicators."""
        indicators = {}
        
        # RSI
        rsi = RSIIndicator(df['Close'], window=min(14, windows['medium']))
        indicators['rsi'] = rsi.rsi().iloc[-1]
        
        # Stochastic
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'],
                                   window=min(14, windows['medium']),
                                   smooth_window=min(3, windows['short']))
        indicators['stoch_k'] = stoch.stoch().iloc[-1]
        indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
        
        # Williams %R - Updated to use lbp parameter instead of window
        willr = WilliamsRIndicator(df['High'], df['Low'], df['Close'],
                                 lbp=min(14, windows['medium']))
        indicators['willr'] = willr.williams_r().iloc[-1]
        
        return indicators

    def _calculate_volume_indicators(self, df: pd.DataFrame, windows: Dict[str, int]) -> Dict[str, float]:
        """Calculate volume-based indicators."""
        indicators = {}
        
        # OBV
        obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
        indicators['obv'] = obv.on_balance_volume().iloc[-1]
        
        # CMF
        cmf = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'],
                                      window=min(20, windows['short']))
        indicators['cmf'] = cmf.chaikin_money_flow().iloc[-1]
        
        # Volume SMA
        indicators['volume_sma_20'] = ta.trend.sma_indicator(df['Volume'], window=20).iloc[-1]
        
        return indicators

    def _calculate_volatility_indicators(self, df: pd.DataFrame, windows: Dict[str, int]) -> Dict[str, float]:
        """Calculate volatility indicators."""
        indicators = {}
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'], window=min(20, windows['short']))
        indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
        indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
        indicators['bb_width'] = bb.bollinger_wband().iloc[-1]
        
        # ATR
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'],
                             window=min(14, windows['short']))
        indicators['atr'] = atr.average_true_range().iloc[-1]
        
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
            df['Swing_High'] = False
            df['Swing_Low'] = False
            
            for i in range(window, len(df) - window):
                # Check for swing high
                if all(df['High'].iloc[i] > df['High'].iloc[i-window:i]) and \
                   all(df['High'].iloc[i] > df['High'].iloc[i+1:i+window+1]):
                    df.loc[df.index[i], 'Swing_High'] = True
                
                # Check for swing low
                if all(df['Low'].iloc[i] < df['Low'].iloc[i-window:i]) and \
                   all(df['Low'].iloc[i] < df['Low'].iloc[i+1:i+window+1]):
                    df.loc[df.index[i], 'Swing_Low'] = True
            
            # Get recent swing points
            recent_highs = df[df['Swing_High']].tail(3)
            recent_lows = df[df['Swing_Low']].tail(3)
            
            if not recent_highs.empty:
                analysis['swing_highs'] = recent_highs['High'].tolist()
            
            if not recent_lows.empty:
                analysis['swing_lows'] = recent_lows['Low'].tolist()
            
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
        highs = recent['High'].values
        lows = recent['Low'].values
        closes = recent['Close'].values
        
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
            if candle['Body_Size'] > 0:  # Not a doji
                if candle['OC_Range'] > 0:  # Bullish candle
                    if candle['Lower_Shadow'] > 2 * candle['Body_Size']:
                        analysis['price_action'].append('hammer')
                else:  # Bearish candle
                    if candle['Upper_Shadow'] > 2 * candle['Body_Size']:
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

    def generate_trade_signals(self) -> bool:
        """Generate swing trading signals for all securities."""
        try:
            # Load risk-scored securities
            data = load_from_json(self.input_file)
            if not data or 'risk_scored_securities' not in data:
                self.logger.error("No securities data found or invalid format")
                return False

            securities = data['risk_scored_securities']
            if not securities:
                self.logger.error("No securities in the list")
                return False

            recommendations = []
            for security in securities:
                ticker = security.get('Symbol') or security.get('symbol')  # Try both cases
                if not ticker:
                    self.logger.warning(f"Missing symbol in security data: {security}")
                    continue
                
                # Fetch data for multiple timeframes
                data = {}
                for timeframe, (period, interval) in self.timeframes.items():
                    df = self._get_stock_data(ticker, period=f"{period}d", interval=interval)
                    if df is not None:
                        data[timeframe] = df

                if not data:
                    self.logger.warning(f"No data available for {ticker}")
                    continue

                # Calculate indicators for each timeframe
                indicators = {}
                for timeframe, df in data.items():
                    timeframe_indicators = self._calculate_technical_indicators(df)
                    if timeframe_indicators:
                        indicators[timeframe] = timeframe_indicators

                if not indicators:
                    continue

                # Determine signal for each timeframe
                signals = {}
                for timeframe, timeframe_indicators in indicators.items():
                    signal, timeframe, justification = self._determine_signal_and_timeframe(timeframe_indicators)
                    signals[timeframe] = {
                        'signal': signal,
                        'timeframe': timeframe,
                        'justification': justification
                    }

                # Determine strongest signal
                strongest_signal = max(signals.items(), 
                                    key=lambda x: abs(self.signal_thresholds['strong'] if x[1]['signal'] != 'HOLD' else 0))

                # Add recommendation
                recommendations.append({
                    'ticker': ticker,
                    'name': security['name'],
                    'signal': strongest_signal[1]['signal'],
                    'timeframe': strongest_signal[1]['timeframe'],
                    'confidence': strongest_signal[1]['justification'][0].split()[0],
                    'price': indicators[strongest_signal[0]]['current_price'],
                    'position_size': indicators[strongest_signal[0]].get('position_size', 0.05),
                    'justification': strongest_signal[1]['justification']
                })

            # Save recommendations
            if recommendations:
                df = pd.DataFrame(recommendations)
                df.to_csv(self.output_file, index=False)
                self.logger.info(f"Saved {len(recommendations)} swing trading recommendations to {self.output_file}")
                return True
            else:
                self.logger.warning("No swing trading recommendations generated")
                return False

        except Exception as e:
            self.logger.error(f"Error generating swing trading signals: {str(e)}")
            return False

    def analyze_security(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single security's historical data to generate trading signals.
        
        Args:
            historical_data: Dictionary containing historical price data and indicators
            
        Returns:
            Dictionary containing trading signals and analysis
        """
        try:
            # Extract the DataFrame from historical data
            df = historical_data.get('history')
            if df is None or df.empty:
                self.logger.warning("No historical data provided for analysis")
                return None
                
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(df)
            if not indicators:
                self.logger.warning("Failed to calculate technical indicators")
                return None
            
            # Determine trading signal
            signal, timeframe, justification = self._determine_signal_and_timeframe(indicators)
            
            # Prepare analysis results
            analysis = {
                'signal': signal,
                'timeframe': timeframe,
                'confidence': justification[0].split()[0],
                'price': indicators['current_price'],
                'position_size': indicators.get('position_size', 0.05),
                'justification': justification,
                'indicators': {
                    'trend': {
                        'adx': indicators['adx'],
                        'macd': indicators['macd'],
                        'macd_signal': indicators['macd_signal']
                    },
                    'momentum': {
                        'rsi': indicators['rsi'],
                        'stoch_k': indicators['stoch_k'],
                        'stoch_d': indicators['stoch_d']
                    },
                    'volume': {
                        'obv': indicators['obv'],
                        'cmf': indicators['cmf']
                    },
                    'volatility': {
                        'atr': indicators['atr'],
                        'bb_width': indicators['bb_width']
                    }
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing security: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None


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