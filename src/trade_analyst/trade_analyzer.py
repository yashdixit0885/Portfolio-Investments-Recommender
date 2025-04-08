import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from utils.common import setup_logging, get_current_time, save_to_json, load_from_json
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange

class TradeAnalyzer:
    def __init__(self):
        self.logger = setup_logging('trade_analyzer')
        self.recommendations_file = 'data/trade_recommendations.json'
        self.analysis_file = 'data/trade_analysis.json'
        self.min_return_threshold = 0.15  # 15% minimum return threshold
        
        # Define swing trading timeframes
        self.timeframes = {
            'short_term': '1mo',  # 1 month for short-term
            'medium_term': '3mo',  # 3 months for medium-term
            'long_term': '6mo'  # 6 months for long-term
        }
        
    def _get_stock_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get historical stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=self.timeframes[timeframe], interval='1d')
            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
                
            self.logger.info(f"Got {len(df)} data points for {symbol} in {timeframe} timeframe")
            self.logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
            
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for the given data"""
        try:
            if len(df) < 26:  # Minimum required periods for Ichimoku
                self.logger.warning("Not enough data points for technical analysis")
                return None
                
            self.logger.debug(f"Calculating technical indicators for {len(df)} data points")
                
            # Initialize indicators with price data
            ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
            bollinger = BollingerBands(close=df['Close'])
            stochastic = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
            macd_ind = MACD(close=df['Close'])
            rsi_ind = RSIIndicator(close=df['Close'])
            
            # Get the indicator values
            tenkan = ichimoku.ichimoku_conversion_line()
            kijun = ichimoku.ichimoku_base_line()
            senkou_a = ichimoku.ichimoku_a()
            senkou_b = ichimoku.ichimoku_b()
            
            self.logger.debug("Ichimoku values:")
            self.logger.debug(f"Tenkan-sen: {tenkan.iloc[-1]:.2f}")
            self.logger.debug(f"Kijun-sen: {kijun.iloc[-1]:.2f}")
            self.logger.debug(f"Senkou Span A: {senkou_a.iloc[-1]:.2f}")
            self.logger.debug(f"Senkou Span B: {senkou_b.iloc[-1]:.2f}")
            
            # Ensure we have valid data
            if tenkan.empty or kijun.empty or senkou_a.empty or senkou_b.empty:
                self.logger.warning("Invalid indicator data")
                return None
                
            # Get Bollinger Bands values
            bb_upper = bollinger.bollinger_hband()
            bb_middle = bollinger.bollinger_mavg()
            bb_lower = bollinger.bollinger_lband()
            
            self.logger.debug("Bollinger Bands values:")
            self.logger.debug(f"Upper: {bb_upper.iloc[-1]:.2f}")
            self.logger.debug(f"Middle: {bb_middle.iloc[-1]:.2f}")
            self.logger.debug(f"Lower: {bb_lower.iloc[-1]:.2f}")
            
            # Get Stochastic values
            stoch_k, stoch_d = self._calculate_stochastic(df)
            
            self.logger.debug("Stochastic values:")
            self.logger.debug(f"%K: {stoch_k:.2f}")
            self.logger.debug(f"%D: {stoch_d:.2f}")
            
            # Get MACD values
            macd = macd_ind.macd()
            macd_signal = macd_ind.macd_signal()
            
            # Get RSI value
            rsi = rsi_ind.rsi()
            
            # Compile all indicators
            indicators = {
                'ichimoku': {
                    'tenkan': tenkan,
                    'kijun': kijun,
                    'senkou_span_a': senkou_a,
                    'senkou_span_b': senkou_b
                },
                'bollinger_bands': {
                    'upper': bb_upper,
                    'middle': bb_middle,
                    'lower': bb_lower
                },
                'stochastic': {
                    '%K': stoch_k,
                    '%D': stoch_d
                },
                'macd': {
                    'macd': macd,
                    'signal': macd_signal
                },
                'rsi': rsi
            }
            
            self.logger.debug("Successfully calculated all technical indicators")
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return None
            
    def analyze_technical_indicators(self, ticker: str, timeframe: str = '1y') -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=timeframe)
            
            # Calculate various technical indicators
            sma_20 = SMAIndicator(close=hist['Close'], window=20)
            sma_50 = SMAIndicator(close=hist['Close'], window=50)
            ema_20 = EMAIndicator(close=hist['Close'], window=20)
            macd = MACD(close=hist['Close'])
            rsi = RSIIndicator(close=hist['Close'])
            vwap = VolumeWeightedAveragePrice(high=hist['High'], low=hist['Low'], 
                                             close=hist['Close'], volume=hist['Volume'])
            
            # Get current values
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
            
            return {
                'current_price': current_price,
                'sma_20': sma_20.sma_indicator().iloc[-1],
                'sma_50': sma_50.sma_indicator().iloc[-1],
                'ema_20': ema_20.ema_indicator().iloc[-1],
                'macd': macd.macd().iloc[-1],
                'macd_signal': macd.macd_signal().iloc[-1],
                'rsi': rsi.rsi().iloc[-1],
                'vwap': vwap.volume_weighted_average_price().iloc[-1],
                'volume_ratio': current_volume / avg_volume,
                'price_vs_sma20': (current_price / sma_20.sma_indicator().iloc[-1] - 1) * 100,
                'price_vs_sma50': (current_price / sma_50.sma_indicator().iloc[-1] - 1) * 100
            }
        except Exception as e:
            self.logger.error(f"Error in technical analysis for {ticker}: {str(e)}")
            return {}

    def analyze_risk_metrics(self, ticker: str) -> Dict[str, Any]:
        """Analyze risk metrics for the trade"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            # Calculate volatility
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate maximum drawdown
            rolling_max = hist['Close'].rolling(window=252, min_periods=1).max()
            drawdowns = hist['Close'] / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'var_95': returns.quantile(0.05),  # 95% Value at Risk
                'var_99': returns.quantile(0.01)   # 99% Value at Risk
            }
        except Exception as e:
            self.logger.error(f"Error in risk analysis for {ticker}: {str(e)}")
            return {}

    def calculate_potential_return(self, ticker: str, position: str) -> Dict[str, Any]:
        """Calculate potential return based on technical analysis"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            # Calculate various price targets
            current_price = hist['Close'].iloc[-1]
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            atr = self._calculate_atr(hist)
            
            if position == 'long':
                # Calculate potential upside targets
                targets = {
                    'conservative': current_price + (atr * 2),
                    'moderate': current_price + (atr * 3),
                    'aggressive': current_price + (atr * 4)
                }
            else:  # short position
                # Calculate potential downside targets
                targets = {
                    'conservative': current_price - (atr * 2),
                    'moderate': current_price - (atr * 3),
                    'aggressive': current_price - (atr * 4)
                }
            
            # Calculate potential returns
            returns = {
                'conservative': abs((targets['conservative'] - current_price) / current_price),
                'moderate': abs((targets['moderate'] - current_price) / current_price),
                'aggressive': abs((targets['aggressive'] - current_price) / current_price)
            }
            
            return {
                'targets': targets,
                'returns': returns,
                'meets_threshold': any(ret >= self.min_return_threshold for ret in returns.values())
            }
        except Exception as e:
            self.logger.error(f"Error calculating potential return for {ticker}: {str(e)}")
            return {}

    def _calculate_atr(self, data: Union[pd.DataFrame, float], bb: Optional[Dict[str, pd.Series]] = None, period: int = 14) -> float:
        """Calculate Average True Range
        
        Args:
            data: Either a DataFrame with OHLC data or the current price
            bb: Optional Bollinger Bands dictionary
            period: Period for ATR calculation, defaults to 14
            
        Returns:
            float: ATR value
        """
        try:
            if isinstance(data, pd.DataFrame):
                high = data['High']
                low = data['Low']
                close = data['Close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=period).mean().iloc[-1]
                
            else:  # data is current_price
                if bb is None or not all(k in bb for k in ['upper', 'lower']):
                    return 0.0
                    
                current_price = float(data)
                band_width = bb['upper'].iloc[-1] - bb['lower'].iloc[-1]
                atr = band_width / 4  # Approximate ATR using BB width
                
            return float(atr)
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0

    def analyze_swing_patterns(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze swing trading patterns for a given symbol and timeframe.
        """
        try:
            data = self.get_historical_data(symbol, timeframe)
            if data is None or len(data) == 0:
                logging.warning(f"No data available for {symbol} in {timeframe} timeframe")
                return {}

            # Get the latest price
            current_price = data['Close'].iloc[-1]
            logging.info(f"Current price for {symbol}: {current_price:.2f}")

            # Calculate technical indicators
            ichimoku = self._calculate_ichimoku(data)
            bb = self._calculate_bollinger_bands(data)
            stoch = self._calculate_stochastic(data)
            
            # Calculate price changes
            price_changes = {
                '1d': self._calculate_change(data['Close'], 1),
                '5d': self._calculate_change(data['Close'], 5),
                '20d': self._calculate_change(data['Close'], 20)
            }

            # Calculate volume changes
            volume_changes = {
                '1d': self._calculate_change(data['Volume'], 1),
                '5d': self._calculate_change(data['Volume'], 5),
                '20d': self._calculate_change(data['Volume'], 20)
            }

            # Identify patterns
            patterns = self._identify_swing_pattern(data, ichimoku, bb, stoch)

            # Prepare analysis results
            analysis_results = {
                'current_price': current_price,
                'price_changes': price_changes,
                'volume_changes': volume_changes,
                'technical_indicators': {
                    'ichimoku': ichimoku,
                    'bollinger_bands': bb,
                    'stochastic': stoch
                },
                'pattern_analysis': patterns
            }

            # Log analysis results
            logging.info(f"Analysis results for {symbol} ({timeframe}):")
            logging.info(f"Price changes: 1d={price_changes['1d']:.2f}%, 5d={price_changes['5d']:.2f}%, 20d={price_changes['20d']:.2f}%")
            logging.info(f"Volume changes: 1d={volume_changes['1d']:.2f}%, 5d={volume_changes['5d']:.2f}%, 20d={volume_changes['20d']:.2f}%")
            
            if patterns:
                logging.info(f"Patterns identified: {patterns}")
            else:
                logging.info("No significant patterns identified")

            return analysis_results

        except Exception as e:
            logging.error(f"Error analyzing swing patterns: {str(e)}")
            return {}

    def _calculate_change(self, series: pd.Series, periods: int) -> float:
        """
        Calculate percentage change over specified periods with validation.
        Returns 0.0 if data is invalid or change is unrealistic.
        """
        try:
            if series is None or series.empty or len(series) <= periods:
                return 0.0
            
            current = series.iloc[-1]
            previous = series.iloc[-periods-1]
            
            if pd.isna(current) or pd.isna(previous) or previous == 0:
                return 0.0
            
            change = ((current - previous) / previous) * 100.0
            
            # Filter out unrealistic changes (>100% for price, >1000% for volume)
            if abs(change) > 100 and 'Price' in str(series.name):
                self.logger.warning(f"Unrealistic price change filtered: {change:.2f}%")
                return 0.0
            elif abs(change) > 1000 and 'Volume' in str(series.name):
                self.logger.warning(f"Unrealistic volume change filtered: {change:.2f}%")
                return 0.0
                
            return change
            
        except Exception as e:
            self.logger.error(f"Error calculating change: {str(e)}")
            return 0.0

    def _identify_swing_pattern(self, data: pd.DataFrame, ichimoku: Dict, bb: Dict, stoch: Dict) -> Dict[str, Any]:
        """
        Identify swing trading patterns based on technical indicators.
        Returns a dictionary with direction, strength, and specific signals.
        """
        try:
            if data is None or data.empty or len(data) < 2:
                return {'direction': 'neutral', 'strength': 0, 'signals': [], 'action': 'HOLD'}

            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            
            if pd.isna(current_price) or pd.isna(prev_price):
                return {'direction': 'neutral', 'strength': 0, 'signals': [], 'action': 'HOLD'}

            patterns = {
                'direction': 'neutral',
                'strength': 0,
                'signals': [],
                'action': 'HOLD'
            }

            # Ichimoku Cloud analysis
            if all(k in ichimoku for k in ['senkou_span_a', 'senkou_span_b', 'tenkan', 'kijun']):
                span_a = ichimoku['senkou_span_a'].iloc[-1] if isinstance(ichimoku['senkou_span_a'], pd.Series) else ichimoku['senkou_span_a']
                span_b = ichimoku['senkou_span_b'].iloc[-1] if isinstance(ichimoku['senkou_span_b'], pd.Series) else ichimoku['senkou_span_b']
                tenkan = ichimoku['tenkan'].iloc[-1] if isinstance(ichimoku['tenkan'], pd.Series) else ichimoku['tenkan']
                kijun = ichimoku['kijun'].iloc[-1] if isinstance(ichimoku['kijun'], pd.Series) else ichimoku['kijun']
                
                if not any(pd.isna(x) for x in [span_a, span_b, tenkan, kijun]):
                    if current_price > max(span_a, span_b):
                        patterns['signals'].append('Above Ichimoku Cloud')
                        patterns['strength'] += 1
                        if tenkan > kijun:  # Strong bullish confirmation
                            patterns['strength'] += 1
                            patterns['signals'].append('Bullish TK Cross')
                    elif current_price < min(span_a, span_b):
                        patterns['signals'].append('Below Ichimoku Cloud')
                        patterns['strength'] -= 1
                        if tenkan < kijun:  # Strong bearish confirmation
                            patterns['strength'] -= 1
                            patterns['signals'].append('Bearish TK Cross')
                    elif span_a > span_b:
                        patterns['signals'].append('Inside Bullish Cloud')
                        patterns['strength'] += 0.5
                    else:
                        patterns['signals'].append('Inside Bearish Cloud')
                        patterns['strength'] -= 0.5

            # Bollinger Bands analysis
            if all(k in bb for k in ['upper', 'lower', 'middle']):
                upper = bb['upper'].iloc[-1] if isinstance(bb['upper'], pd.Series) else bb['upper']
                lower = bb['lower'].iloc[-1] if isinstance(bb['lower'], pd.Series) else bb['lower']
                middle = bb['middle'].iloc[-1] if isinstance(bb['middle'], pd.Series) else bb['middle']
                
                if not any(pd.isna(x) for x in [upper, lower, middle]):
                    band_width = (upper - lower) / middle
                    
                    if current_price > upper:
                        patterns['signals'].append('Above Upper Bollinger')
                        patterns['strength'] -= 0.5
                    elif current_price < lower:
                        patterns['signals'].append('Below Lower Bollinger')
                        patterns['strength'] += 0.5
                    
                    # Bollinger squeeze detection
                    if band_width < 0.1:
                        patterns['signals'].append('Bollinger Squeeze')
                        if current_price > middle:
                            patterns['strength'] += 0.5
                        else:
                            patterns['strength'] -= 0.5

            # Stochastic analysis
            if all(k in stoch for k in ['%K', '%D']):
                k_value = stoch['%K'].iloc[-1] if isinstance(stoch['%K'], pd.Series) else stoch['%K']
                d_value = stoch['%D'].iloc[-1] if isinstance(stoch['%D'], pd.Series) else stoch['%D']
                
                if not pd.isna(k_value) and not pd.isna(d_value):
                    if k_value > 80:
                        if d_value > 80:
                            patterns['signals'].append('Stochastic Overbought')
                            patterns['strength'] -= 0.5
                    elif k_value < 20:
                        if d_value < 20:
                            patterns['signals'].append('Stochastic Oversold')
                            patterns['strength'] += 0.5
                    
                    if k_value > d_value:
                        if k_value < 80:  # Bullish crossover not in overbought
                            patterns['signals'].append('Stochastic Bullish Cross')
                            patterns['strength'] += 0.5
                    elif k_value < d_value:
                        if k_value > 20:  # Bearish crossover not in oversold
                            patterns['signals'].append('Stochastic Bearish Cross')
                            patterns['strength'] -= 0.5

            # Determine direction and action based on strength
            if patterns['strength'] >= 1:
                patterns['direction'] = 'strong_bullish'
                patterns['action'] = 'BUY'
            elif patterns['strength'] > 0:
                patterns['direction'] = 'bullish'
                patterns['action'] = 'BUY'
            elif patterns['strength'] <= -1:
                patterns['direction'] = 'strong_bearish'
                patterns['action'] = 'SELL'
            elif patterns['strength'] < 0:
                patterns['direction'] = 'bearish'
                patterns['action'] = 'SELL'
            else:
                patterns['direction'] = 'neutral'
                patterns['action'] = 'HOLD'

            return patterns

        except Exception as e:
            self.logger.error(f"Error identifying patterns: {str(e)}")
            return {'direction': 'neutral', 'strength': 0, 'signals': [], 'action': 'HOLD'}

    def calculate_setup_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate a setup score based on technical indicators"""
        try:
            score = 0.0
            
            # Price momentum
            price_changes = analysis.get('price_changes', {})
            if price_changes:
                if price_changes.get('1d', 0) > 0:
                    score += 0.1
                if price_changes.get('5d', 0) > 0:
                    score += 0.15
                if price_changes.get('20d', 0) > 0:
                    score += 0.2
            
            # Volume confirmation
            volume_changes = analysis.get('volume_changes', {})
            if volume_changes:
                if volume_changes.get('1d', 0) > 20:
                    score += 0.1
                if volume_changes.get('5d', 0) > 30:
                    score += 0.15
            
            # Pattern strength
            patterns = analysis.get('pattern_analysis', {})
            if patterns:
                strength = patterns.get('strength', 0)
                score += (strength * 0.2)  # Weight pattern strength
                
                # Additional score for specific signals
                signals = patterns.get('signals', [])
                if 'Bollinger Squeeze' in signals:
                    score += 0.1
                if any('Cross' in signal for signal in signals):
                    score += 0.15
            
            return min(max(score, 0.0), 1.0)  # Normalize between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating setup score: {str(e)}")
            return 0.0
            
    def get_trade_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get trade parameters based on technical analysis"""
        try:
            current_price = analysis['current_price']
            bb = analysis['bollinger_bands']
            
            # Get the latest valid values
            upper = bb['upper'].dropna().iloc[-1] if not bb['upper'].empty else current_price
            lower = bb['lower'].dropna().iloc[-1] if not bb['lower'].empty else current_price
            
            # Determine entry type based on indicators
            if current_price > upper:
                entry_type = 'SHORT'
            elif current_price < lower:
                entry_type = 'LONG'
            else:
                entry_type = 'NEUTRAL'
                
            # Calculate stop loss and take profit levels
            atr = self._calculate_atr(current_price, bb)
            stop_loss = current_price - (2 * atr) if entry_type == 'LONG' else current_price + (2 * atr)
            take_profit = current_price + (3 * atr) if entry_type == 'LONG' else current_price - (3 * atr)
            
            # Determine position size and risk level
            position_size = 5.0  # Default 5% of portfolio
            risk_level = self._calculate_risk_level(analysis)
            
            return {
                'entry_type': entry_type,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'risk_level': risk_level
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trade parameters: {str(e)}")
            return None
            
    def _calculate_risk_level(self, analysis: Dict[str, Any]) -> str:
        """Calculate risk level based on technical indicators"""
        try:
            # Get the latest valid values
            stoch = analysis['stochastic']
            bb = analysis['bollinger_bands']
            price = analysis['current_price']
            
            k = stoch['k'].dropna().iloc[-1] if not stoch['k'].empty else 50
            upper = bb['upper'].dropna().iloc[-1] if not bb['upper'].empty else price
            lower = bb['lower'].dropna().iloc[-1] if not bb['lower'].empty else price
            
            if (k > 80 and price > upper) or (k < 20 and price < lower):
                return 'HIGH'
            elif lower < price < upper:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {str(e)}")
            return 'MEDIUM'
            
    def make_trade_decision(self, analysis: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Make final trade decision based on analysis and parameters"""
        try:
            score = self.calculate_setup_score(analysis)
            confidence = abs(score)
            
            reasons = []
            action = 'HOLD'
            timeframe = analysis.get('timeframe', 'medium_term')  # Get timeframe from analysis
            
            # Lower threshold from 0.3 to 0.2 for more opportunities
            if score > 0.2:
                action = 'BUY'
                reasons.append(f"Bullish technical setup ({timeframe})")
                
                # Get latest stochastic K value
                k = analysis['stochastic']['k'].iloc[-1] if not analysis['stochastic']['k'].empty else 50
                if k < 30:  # Increased from 20 to 30
                    reasons.append("Oversold conditions")
                if params['risk_level'] == 'LOW':
                    reasons.append("Favorable risk profile")
                    
            elif score < -0.2:  # Changed from -0.3 to -0.2
                action = 'SELL'
                reasons.append(f"Bearish technical setup ({timeframe})")
                
                # Get latest stochastic K value
                k = analysis['stochastic']['k'].iloc[-1] if not analysis['stochastic']['k'].empty else 50
                if k > 70:  # Decreased from 80 to 70
                    reasons.append("Overbought conditions")
                if params['risk_level'] == 'HIGH':
                    reasons.append("High risk profile")
                    
            else:
                reasons.append(f"No clear directional bias ({timeframe})")
                reasons.append("Wait for stronger setup")
                
            return {
                'action': action,
                'timeframe': timeframe,
                'confidence': confidence,
                'reasons': reasons,
                'risk_level': params['risk_level'],
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit']
            }
            
        except Exception as e:
            self.logger.error(f"Error making trade decision: {str(e)}")
            return {
                'action': 'HOLD',
                'timeframe': 'unknown',
                'confidence': 0.0,
                'reasons': ['Error analyzing trade setup'],
                'risk_level': 'MEDIUM',
                'stop_loss': None,
                'take_profit': None
            }

    def analyze_timeframe_opportunity(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Analyze trading opportunity for specific timeframe"""
        try:
            # Get swing patterns for the timeframe
            swing_analysis = self.analyze_swing_patterns(ticker, timeframe)
            
            # Get technical indicators
            technical = self.analyze_technical_indicators(ticker, timeframe)
            
            # Get risk metrics
            risk = self.analyze_risk_metrics(ticker)
            
            # Calculate potential return
            potential = self.calculate_potential_return(ticker, 'long' if technical.get('rsi', 50) < 70 else 'short')
            
            # Determine if this is a good swing trade setup
            setup_score = self.calculate_setup_score(swing_analysis)
            
            return {
                'timeframe': timeframe,
                'swing_analysis': swing_analysis,
                'technical_analysis': technical,
                'risk_metrics': risk,
                'potential_returns': potential,
                'setup_score': setup_score,
                'trade_setup': self.get_trade_parameters(swing_analysis)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe opportunity for {ticker}: {str(e)}")
            return {}

    def evaluate_trade(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a trade recommendation across multiple timeframes"""
        ticker = recommendation['ticker']
        
        # Analyze each timeframe
        timeframe_analyses = {}
        for timeframe, period in self.timeframes.items():
            timeframe_analyses[timeframe] = self.analyze_timeframe_opportunity(ticker, period)
            
        # Find best timeframe for swing trade
        best_timeframe = max(timeframe_analyses.items(), 
                           key=lambda x: x[1].get('setup_score', 0))[0]
        
        # Combine all analyses
        analysis = {
            'ticker': ticker,
            'original_recommendation': recommendation,
            'timeframe_analyses': timeframe_analyses,
            'best_timeframe': best_timeframe,
            'best_setup': timeframe_analyses[best_timeframe].get('trade_setup', {}),
            'timestamp': get_current_time().isoformat(),
            'trade_decision': self.make_trade_decision(
                timeframe_analyses[best_timeframe].get('swing_analysis', {}),
                timeframe_analyses[best_timeframe].get('technical_analysis', {}),
                timeframe_analyses[best_timeframe].get('risk_metrics', {})
            )
        }
        
        # Save analysis
        self._save_analysis(analysis)
        
        return analysis

    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save trade analysis results"""
        try:
            existing_analyses = load_from_json(self.analysis_file)
            if not isinstance(existing_analyses, list):
                existing_analyses = []
            
            existing_analyses.append(analysis)
            save_to_json(existing_analyses, self.analysis_file)
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")

    def process_recommendations(self) -> None:
        """Process all trade recommendations"""
        try:
            recommendations = load_from_json(self.recommendations_file)
            if not recommendations:
                self.logger.info("No new recommendations to analyze")
                return
                
            for recommendation in recommendations:
                self.logger.info(f"Analyzing swing trade opportunity for {recommendation['ticker']}")
                analysis = self.evaluate_trade(recommendation)
                
                if analysis['trade_decision']['action'] == 'execute':
                    self.logger.info(f"Swing trade approved for {analysis['ticker']}")
                    self.logger.info(f"Best timeframe: {analysis['best_timeframe']}")
                    self.logger.info(f"Setup type: {analysis['best_setup']['entry_type']}")
                    
        except Exception as e:
            self.logger.error(f"Error processing recommendations: {str(e)}")

    def _calculate_support_levels(self, data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Calculate support and resistance levels using rolling min/max.
        
        Args:
            data: DataFrame containing price data
            window: Window size for rolling calculations
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        try:
            # Calculate rolling min/max
            rolling_min = data['Low'].rolling(window=window).min()
            rolling_max = data['High'].rolling(window=window).max()
            
            # Get the most recent support/resistance levels
            support_level = rolling_min.iloc[-1]
            resistance_level = rolling_max.iloc[-1]
            
            return float(support_level), float(resistance_level)
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance levels: {str(e)}")
            return 0.0, 0.0

    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Bollinger Bands indicators.
        """
        try:
            close = data['Close']
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            
            upper = sma20 + (std20 * 2)
            lower = sma20 - (std20 * 2)
            
            return {
                'upper': float(upper.iloc[-1]),
                'middle': float(sma20.iloc[-1]),
                'lower': float(lower.iloc[-1])
            }
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0
            }

    def _calculate_stochastic(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Stochastic Oscillator indicators.
        """
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # Calculate %K
            period14_high = high.rolling(window=14).max()
            period14_low = low.rolling(window=14).min()
            k = 100 * ((close - period14_low) / (period14_high - period14_low))
            
            # Calculate %D (3-day SMA of %K)
            d = k.rolling(window=3).mean()
            
            return {
                'k': float(k.iloc[-1]),
                'd': float(d.iloc[-1])
            }
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            return {
                'k': 0.0,
                'd': 0.0
            }

    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Ichimoku Cloud indicators.
        """
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # Calculate Tenkan-sen (Conversion Line)
            period9_high = high.rolling(window=9).max()
            period9_low = low.rolling(window=9).min()
            tenkan = (period9_high + period9_low) / 2
            
            # Calculate Kijun-sen (Base Line)
            period26_high = high.rolling(window=26).max()
            period26_low = low.rolling(window=26).min()
            kijun = (period26_high + period26_low) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan + kijun) / 2).shift(26)
            
            # Calculate Senkou Span B (Leading Span B)
            period52_high = high.rolling(window=52).max()
            period52_low = low.rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            return {
                'tenkan': float(tenkan.iloc[-1]),
                'kijun': float(kijun.iloc[-1]),
                'senkou_span_a': float(senkou_span_a.iloc[-1]),
                'senkou_span_b': float(senkou_span_b.iloc[-1])
            }
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            return {
                'tenkan': 0.0,
                'kijun': 0.0,
                'senkou_span_a': 0.0,
                'senkou_span_b': 0.0
            }

    def get_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch historical data for a given symbol and timeframe using yfinance.
        
        Args:
            symbol (str): The stock symbol to fetch data for
            timeframe (str): The timeframe to fetch data for ('short_term', 'medium_term', 'long_term')
            
        Returns:
            pd.DataFrame: Historical price data or None if data fetch fails
        """
        try:
            # Map timeframes to yfinance periods
            timeframe_map = {
                'short_term': '5d',
                'medium_term': '3mo',
                'long_term': '6mo'
            }
            
            period = timeframe_map.get(timeframe)
            if not period:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            # Fetch data using yfinance
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval='1d')
            
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

if __name__ == "__main__":
    analyzer = TradeAnalyzer()
    analyzer.process_recommendations()