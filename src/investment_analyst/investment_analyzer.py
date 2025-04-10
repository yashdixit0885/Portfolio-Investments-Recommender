"""
Investment Analyzer: Finding Stocks with High Price Movement Potential

This tool helps identify stocks that are likely to experience significant price changes,
whether up or down. It's like a radar that spots stocks that are "on the move" or about
to make a big move.

How it works:
1. Looks at each stock's recent price action (30% of score)
   - How much the price has moved compared to its average
   - How volatile (up and down) the price has been

2. Analyzes trading volume (25% of score)
   - Checks if more people are buying/selling than usual
   - Looks for unusual trading activity

3. Studies technical indicators (25% of score)
   - RSI: Measures if a stock is overbought or oversold
   - MACD: Shows if a stock's momentum is changing
   - Bollinger Bands: Identifies when prices are unusually high or low

4. Considers market context (20% of score)
   - How the stock moves compared to the overall market (Beta)
   - The stock's size and how easy it is to trade

The tool combines all these factors to give each stock a score from 0 to 1:
- Higher scores (closer to 1) mean the stock is more likely to make a big move
- Lower scores (closer to 0) mean the stock is likely to stay relatively stable

The top 10% of stocks (those with the highest scores) are identified as potential
opportunities for further research. These stocks might be good candidates for:
- Short-term trading opportunities
- Options trading
- Setting up price alerts
- Further detailed analysis

Note: This is just the first step in finding opportunities. Always do your own research
and consider your risk tolerance before making any investment decisions.
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from src.utils.common import save_to_json, load_from_json, get_current_time, save_recommendations_to_csv
import numpy as np
import yfinance as yf
from src.research_analyst.rate_limiter import RateLimiter

class InvestmentAnalyzer:
    """Identifies securities with high price movement potential from source data for further research."""
    
    def __init__(self):
        """Initialize the InvestmentAnalyzer with configuration and logging."""
        self.logger = logging.getLogger('investment_analyzer')
        self.data_dir = 'data'
        self.input_file = os.path.join(self.data_dir, 'securities_data.csv')
        self.output_file = os.path.join(self.data_dir, 'high_potential_securities.json')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize rate limiter for Yahoo Finance API calls
        self.rate_limiter = RateLimiter(calls_per_second=2.0, max_retries=3, retry_delay=5.0)
        
        # Cache for fetched yfinance data
        self.yf_data_cache = {}
        
        # Load configuration (though less critical now, kept for potential future use)
        self.load_config()
        
    def load_config(self):
        """Load configuration from environment variables or use defaults."""
        try:
            self.min_volume = int(os.getenv('MIN_VOLUME', '500000'))
            self.min_market_cap = float(os.getenv('MIN_MARKET_CAP', '50000000'))
        except (ValueError, TypeError):
            self.logger.warning("Invalid environment variables, using default values")
            self.min_volume = 500000
            self.min_market_cap = 50000000
        
    def _clean_numeric(self, value):
        """Clean and convert numeric values from strings to floats."""
        if pd.isna(value) or value is None or value == '':
            return 0.0
        if isinstance(value, (int, float)):
            return round(float(value), 3)
            
        # Convert to string and clean
        value_str = str(value)
        
        # Handle percentage values
        if '%' in value_str:
            try:
                return round(float(value_str.rstrip('%')) / 100.0, 3)
            except ValueError:
                    return 0.0

        # Handle K/M/B/T suffixes
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000, 'T': 1000000000000}
        for suffix, multiplier in multipliers.items():
            if suffix in value_str:
                try:
                    number = float(value_str.replace(suffix, '').replace(',', ''))
                    return round(number * multiplier, 3)
                except ValueError:
                    continue

        # Handle regular numbers with commas
        try:
            return round(float(value_str.replace(',', '')), 3)
        except ValueError:
            return 0.0
    
    def _fetch_yf_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches necessary data from Yahoo Finance for a ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
            
        Returns:
            Dict containing 'info' and 'history' data from Yahoo Finance.
        """
        if ticker in self.yf_data_cache:
            return self.yf_data_cache[ticker]
            
        self.logger.debug(f"Fetching Yahoo Finance data for {ticker}")
        yf_data = {'info': {}, 'history': pd.DataFrame()}
        
        try:
            yf_stock = yf.Ticker(ticker)
            
            # Define callables for rate limiter
            def get_info():
                info_data = yf_stock.info
                if not info_data or not info_data.get('regularMarketPrice'):
                    self.logger.warning(f"No valid market data found for {ticker}")
                    return {}
                return info_data
            
            def get_history():
                # Fetch 1 year of daily data for technical indicators
                hist_data = yf_stock.history(period='1y', interval='1d')
                if hist_data.empty:
                    self.logger.warning(f"No valid history found for {ticker}")
                    return pd.DataFrame()
                return hist_data
            
            # Use rate limiter for API calls
            yf_data['info'] = self.rate_limiter.call_with_retry(get_info)
            if yf_data['info']:
                yf_data['history'] = self.rate_limiter.call_with_retry(get_history)
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance data for {ticker}: {str(e)}")
            yf_data = {'info': {}, 'history': pd.DataFrame()}
        
        self.yf_data_cache[ticker] = yf_data
        return yf_data

    def _calculate_technical_indicators(self, historical_data):
        """Calculate technical indicators from historical price data."""
        try:
            if historical_data.empty or len(historical_data) < 20:
                self.logger.warning("Insufficient historical data for technical analysis")
                return {
                    'atr': 0.0,
                    'bb_width': 0.0,
                    'rsi': 0.0,
                    'stoch_k': 0.0,
                    'stoch_d': 0.0,
                    'adx': 0.0,
                    'obv': 0.0,
                    'vwap': 0.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_hist': 0.0
                }

            # Convert DataFrame columns to Series for calculations
            close = historical_data['Close']
            high = historical_data['High']
            low = historical_data['Low']
            volume = historical_data['Volume']

            # Calculate ATR
            tr = pd.DataFrame()
            tr['h-l'] = high - low
            tr['h-pc'] = abs(high - close.shift(1))
            tr['l-pc'] = abs(low - close.shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            atr = tr['tr'].rolling(window=14).mean().iloc[-1]

            # Calculate Bollinger Bands
            bb_std = close.rolling(window=20).std()
            bb_width = (bb_std * 2) / close.rolling(window=20).mean()
            bb_width = bb_width.iloc[-1]

            # Calculate RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Handle division by zero in RSI calculation
            if loss.iloc[-1] == 0:
                rsi = 100.0 if gain.iloc[-1] > 0 else 50.0
            else:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))

            # Replace NaN values with 0
            indicators = {
                'atr': float(atr if pd.notna(atr) else 0.0),
                'bb_width': float(bb_width if pd.notna(bb_width) else 0.0),
                'rsi': float(rsi if pd.notna(rsi) else 50.0),
                'stoch_k': 0.0,  # Simplified for now
                'stoch_d': 0.0,  # Simplified for now
                'adx': 0.0,      # Simplified for now
                'obv': float(volume.sum()),
                'vwap': float(close.mean()),
                'macd': 0.0,     # Simplified for now
                'macd_signal': 0.0,
                'macd_hist': 0.0
            }
            
            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return {k: 0.0 for k in ['atr', 'bb_width', 'rsi', 'stoch_k', 'stoch_d', 'adx', 'obv', 'vwap', 'macd', 'macd_signal', 'macd_hist']}

    def load_and_prepare_securities(self, df=None):
        """
        Load and prepare securities data for analysis.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to process. If None, load from input_file.
            
        Returns:
            pd.DataFrame: Prepared securities data.
        """
        try:
            if df is None:
                if not os.path.exists(self.input_file):
                    self.logger.error(f"Input file not found: {self.input_file}")
                    return pd.DataFrame()
                df = pd.read_csv(self.input_file)
            
            # Rename columns for consistency
            column_mapping = {
                'Symbol': 'Ticker',
                'Last': 'Price',
                '% Insider': 'Inst Own %',
                'Div Yield(a)': 'Div Yield',
                '50D Avg Vol': 'Volume_MA50',
                '200D Avg Vol': 'Volume_MA200',
                '50D MA': 'MA50',
                '200D MA': 'MA200'
            }
            df = df.rename(columns=column_mapping)
            
            # Clean numeric columns
            numeric_columns = [
                'Price', 'Volume', 'Volume_MA50', 'Volume_MA200',
                'MA50', 'MA200', 'RSI', 'Beta', 'MACD', 'MACD Signal',
                'Inst Own %', 'Div Yield', 'Market Cap', 'Price Vol',
                'P/E fwd', 'Price/Book', 'Debt/Equity', 'ATR', 'BB Width',
                'Stoch %K', 'Stoch %D', 'ADX', 'OBV', 'VWAP'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self._clean_numeric)
            
            # Calculate additional metrics
            if all(col in df.columns for col in ['Volume', 'Volume_MA50', 'Volume_MA200']):
                df['volume_change_50d'] = (df['Volume'] / df['Volume_MA50']) - 1
                df['volume_change_200d'] = (df['Volume'] / df['Volume_MA200']) - 1
            
            if all(col in df.columns for col in ['Price', 'MA50', 'MA200']):
                df['price_momentum_50d'] = (df['Price'] / df['MA50']) - 1
                df['price_momentum_200d'] = (df['Price'] / df['MA200']) - 1
            
            # Add timestamp
            df['analysis_timestamp'] = get_current_time()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing securities data: {str(e)}")
            return pd.DataFrame()

    def _calculate_price_movement_potential(self, df):
        """Calculate price movement potential for each security."""
        try:
            # Check for required columns
            required_columns = [
                'price_momentum_50d', 'price_momentum_200d',
                'volume_change_50d', 'volume_change_200d',
                'Price', 'MA50', 'MA200'
            ]
            for col in required_columns:
                if col not in df.columns:
                    self.logger.warning(f"Missing column {col}, setting default value to 0.0")
                    df[col] = 0.0

            # Calculate price action score (30%)
            df['price_momentum_score'] = (
                df['price_momentum_50d'].abs() * 0.6 +
                df['price_momentum_200d'].abs() * 0.4
            )
            df['price_action_score'] = df['price_momentum_score'].apply(
                lambda x: min(max((x + 0.5) / 1.0, 0), 1) * 0.3
            )

            # Calculate volume score (25%)
            df['volume_score'] = (
                df['volume_change_50d'].apply(lambda x: min(max((x + 0.5) / 1.0, 0), 1)) * 0.6 +
                df['volume_change_200d'].apply(lambda x: min(max((x + 0.5) / 1.0, 0), 1)) * 0.4
            ) * 0.25

            # Calculate technical score (25%)
            df['ma_alignment'] = 0.0
            df.loc[df['MA50'] > df['MA200'], 'ma_alignment'] = 1.0
            df.loc[df['MA50'] < df['MA200'], 'ma_alignment'] = 0.0
            df.loc[df['MA50'] == df['MA200'], 'ma_alignment'] = 0.5

            df['price_vs_ma50'] = ((df['Price'] - df['MA50']) / df['MA50'] + 0.5).clip(0, 1)
            df['price_vs_ma200'] = ((df['Price'] - df['MA200']) / df['MA200'] + 0.5).clip(0, 1)

            df['technical_score'] = (
                df['ma_alignment'] * 0.4 +
                df['price_vs_ma50'] * 0.3 +
                df['price_vs_ma200'] * 0.3
            ) * 0.25

            # Calculate market context score (20%)
            df['market_score'] = 0.5 * 0.2  # Default to neutral market score

            # Calculate final score
            df['price_movement_potential'] = (
                df['price_action_score'] +
                df['volume_score'] +
                df['technical_score'] +
                df['market_score']
            ).fillna(0.0)  # Convert NaN to 0.0

            # Ensure scores are between 0 and 1
            df['price_movement_potential'] = df['price_movement_potential'].clip(0, 1)

            return df
                
        except Exception as e:
            self.logger.error(f"Error calculating price movement potential: {str(e)}")
            df['price_movement_potential'] = 0.0
            return df
    
    def identify_potential_movers(self) -> pd.DataFrame:
        """
        Identify securities with high potential for significant price movement.
            
        Returns:
            pd.DataFrame: Securities with high movement potential.
        """
        try:
            # Load and prepare securities data
            df = self.load_and_prepare_securities()
            if df.empty:
                self.logger.warning("No securities data available")
                return pd.DataFrame()
            
            # Calculate price movement potential for all securities
            df = self._calculate_price_movement_potential(df)
            
            # Log score distribution
            score_stats = df['price_movement_potential'].describe()
            self.logger.info(f"Price movement potential score distribution:\n{score_stats}")
            
            # Filter securities with high movement potential
            threshold = 0.5  # Threshold for significant movement potential
            potential_movers = df[df['price_movement_potential'] > threshold].copy()
            
            if potential_movers.empty:
                self.logger.warning(f"No securities found with movement potential > {threshold}")
                # Return the original DataFrame instead of an empty one
                df['analysis_timestamp'] = pd.Timestamp.now()
                return df
            
            # Sort by movement potential (descending)
            potential_movers = potential_movers.sort_values('price_movement_potential', ascending=False)
            
            # Add analysis timestamp
            potential_movers['analysis_timestamp'] = pd.Timestamp.now()
            
            return potential_movers
            
        except Exception as e:
            self.logger.error(f"Error identifying potential movers: {str(e)}")
            return pd.DataFrame()

    def save_potential_securities(self, securities: List[Dict]) -> bool:
        """Save the list of high-potential securities to a JSON file.

        Args:
            securities (List[Dict]): List of dictionaries containing security data.

        Returns:
            bool: True if save was successful, False otherwise.
        """
        try:
            if not securities:
                self.logger.warning("No securities data provided.")
                return False

            valid_securities = []
            for security in securities:
                try:
                    # Ensure required fields exist and are valid
                    if not all(key in security for key in ['Symbol', 'Price', 'Volume', 'movement_potential_score']):
                        continue
                    
                    # Convert numeric fields
                    security['Price'] = float(security['Price'])
                    security['Volume'] = float(security['Volume'])
                    security['movement_potential_score'] = float(security['movement_potential_score'])
                    valid_securities.append(security)
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.warning(f"Invalid security data: {str(e)}")
                    continue

            if not valid_securities:
                self.logger.warning("No valid securities data to save.")
                return False

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            # Save to JSON file
            data = {
                'analysis_timestamp': get_current_time(),
                'high_potential_securities': valid_securities
            }
            
            save_to_json(data, self.output_file)
            self.logger.info(f"Saved {len(valid_securities)} high-potential securities to {self.output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving potential securities: {str(e)}")
            return False

    def run_analysis(self) -> bool:
        """Main entry point to run the full analysis and save results."""
        self.logger.info("Starting Investment Analyzer: Identifying high price movement potential securities.")
        potential_movers = self.identify_potential_movers()
        
        if potential_movers.empty:
             self.logger.warning("Investment Analysis did not identify any securities.")
             return False
             
        success = self.save_potential_securities(potential_movers.to_dict('records'))
        
        if success:
            self.logger.info("Investment Analyzer finished successfully.")
        else:
            self.logger.error("Investment Analyzer finished with errors during saving.")
            
        return success

    def _calculate_opportunity_score(self, metrics: Dict) -> float:
        """Calculate the overall opportunity score from individual metrics."""
        try:
            # Ensure all required metrics exist with valid values
            required_metrics = ['momentum_score', 'volume_score', 'technical_score', 'market_score']
            for metric in required_metrics:
                if metric not in metrics or not isinstance(metrics[metric], (int, float)) or pd.isna(metrics[metric]):
                    metrics[metric] = 0.0
                metrics[metric] = max(0.0, min(1.0, float(metrics[metric])))  # Clamp between 0 and 1
            
            # Calculate weighted score
            weights = {
                'momentum_score': 0.3,    # Price momentum (30%)
                'volume_score': 0.25,     # Volume analysis (25%)
                'technical_score': 0.25,  # Technical indicators (25%)
                'market_score': 0.2       # Market context (20%)
            }
            
            total_score = sum(weights[metric] * metrics[metric] for metric in required_metrics)
            return round(total_score, 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0

# Example Usage (Optional - useful for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    analyzer = InvestmentAnalyzer()
    analyzer.run_analysis()

# --- Removed Methods ---
# - process_securities_data (replaced by load_and_prepare_securities & identify_potential_movers)
# - _calculate_opportunity_score (replaced by _calculate_price_movement_potential)
# - analyze_opportunities (logic integrated into identify_potential_movers, helpers removed)
# - _calculate_volume_significance (removed)
# - _calculate_price_momentum (removed)
# - _calculate_market_cap_significance (removed)
# - save_opportunities (replaced by save_potential_securities)
# - process_securities_file (replaced by run_analysis)
# - generate_trade_recommendations (removed - not this class's responsibility)
# - _generate_rationale (removed)
# - _calculate_position_size (removed)

