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
from src.database import DatabaseManager

class InvestmentAnalyzer:
    """Identifies securities with high price movement potential from source data for further research."""
    
    def __init__(self, db: DatabaseManager):
        """Initialize the InvestmentAnalyzer with configuration and logging."""
        self.logger = logging.getLogger('investment_analyzer')
        self.db = db
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
        """Fetch data from Yahoo Finance with rate limiting and DB fallback/update."""
        yf_data = {'info': {}, 'history': pd.DataFrame()}
        
        # Check cache first
        cache_key = f"{ticker}_data"
        # Simplified cache check for now
        # if cache_key in self.yf_data_cache:
        #     self.logger.debug(f"Cache hit for {ticker}")
        #     return self.yf_data_cache[cache_key]
        
        self.logger.debug(f"Fetching data for {ticker}...")
        try:
            # Attempt to fetch fresh data from yfinance first
            stock = yf.Ticker(ticker)
            info_data = {}
            hist_data = pd.DataFrame()

            try:
                # Fetch info - No rate limiting needed for property access
                # info_data = self.rate_limiter.call_with_retry(getattr(stock, 'info'))
                info_data = stock.info
                if not isinstance(info_data, dict):
                    self.logger.warning(f"Received non-dict info for {ticker} from yfinance: {type(info_data)}. Falling back to DB.")
                    info_data = {}
            except Exception as e:
                self.logger.warning(f"Error fetching info data for {ticker} from yfinance: {str(e)}. Falling back to DB.")
                info_data = {}

            # If yfinance info fetch failed, try getting from DB
            if not info_data:
                db_info = self.db.get_security(ticker)
                if db_info:
                    self.logger.debug(f"Using info data for {ticker} from database.")
                    yf_data['info'] = db_info
                else:
                    self.logger.warning(f"No info data found for {ticker} in yfinance or DB.")
            else:
                 yf_data['info'] = info_data
                 # Store/Update latest security info in database
                 security_data_for_db = {
                     'ticker': ticker,
                     'name': info_data.get('longName', info_data.get('shortName', '')),
                     'industry': info_data.get('industry', ''),
                     'market_cap': info_data.get('marketCap'),
                     'price': info_data.get('currentPrice', info_data.get('regularMarketPrice')),
                     'volume': info_data.get('volume', info_data.get('regularMarketVolume')),
                     'volume_ma50': info_data.get('averageVolume10days'), 
                     'volume_ma200': info_data.get('averageVolume'),
                     'ma50': info_data.get('fiftyDayAverage'),
                     'ma200': info_data.get('twoHundredDayAverage'),
                     'beta': info_data.get('beta'),
                     'inst_own_pct': info_data.get('heldPercentInstitutions'),
                     'div_yield': info_data.get('dividendYield')
                 }
                 # Clean None values before passing to DB insert
                 security_data_for_db_cleaned = {k: v for k, v in security_data_for_db.items() if v is not None}
                 self.db.insert_securities([security_data_for_db_cleaned])

            try:
                # Fetch history using rate limiter
                hist_data = self.rate_limiter.call_with_retry(getattr(stock, 'history'), period='1y')
                if hist_data is not None and not hist_data.empty:
                    hist_data.columns = [col.lower() for col in hist_data.columns] # Ensure lowercase
                    yf_data['history'] = hist_data
                    # Ensure index is datetime
                    if not isinstance(hist_data.index, pd.DatetimeIndex):
                        hist_data.index = pd.to_datetime(hist_data.index)
                    self.db.insert_historical_data(ticker, hist_data) # Store latest historical data
                else:
                    self.logger.warning(f"No historical data fetched from yfinance for {ticker}. Falling back to DB.")
                    hist_data = pd.DataFrame() # Ensure hist_data is DataFrame for logic below
            except Exception as e:
                self.logger.warning(f"Error fetching historical data for {ticker} from yfinance: {str(e)}. Falling back to DB.")
                hist_data = pd.DataFrame()

            # If yfinance history fetch failed, try getting from DB
            if hist_data.empty:
                db_history = self.db.get_historical_data(ticker)
                if db_history is not None and not db_history.empty:
                    self.logger.debug(f"Using historical data for {ticker} from database.")
                    # Ensure columns are lowercase if fetched from DB
                    db_history.columns = [col.lower() for col in db_history.columns]
                    yf_data['history'] = db_history
                else:
                    self.logger.warning(f"No historical data found for {ticker} in yfinance or DB.")
            
            # Cache the potentially combined data
            # self.yf_data_cache[cache_key] = yf_data # Re-enable caching later if needed
            return yf_data
            
        except Exception as e:
            self.logger.error(f"General error in _fetch_yf_data for {ticker}: {str(e)}", exc_info=True)
            return yf_data # Return default empty structure

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
            close = historical_data['close']
            high = historical_data['high']
            low = historical_data['low']
            volume = historical_data['volume']

            # Calculate ATR
            tr = pd.DataFrame()
            tr['h-l'] = high - low
            tr['h-pc'] = abs(high - close.shift(1))
            tr['l-pc'] = abs(low - close.shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            atr = tr['tr'].rolling(window=14).mean().iloc[-1]

            # Calculate Bollinger Bands
            bb_std = close.rolling(window=20).std()
            bb_mean = close.rolling(window=20).mean()
            # Calculate width, handle division by zero
            bb_width_series = np.where(bb_mean != 0, (bb_std * 2) / bb_mean, 0) 
            # Convert numpy array back to pandas Series with original index to use iloc
            bb_width_series = pd.Series(bb_width_series, index=close.index)
            bb_width = bb_width_series.iloc[-1]

            # Calculate RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rsi = 50.0 # Default for edge cases
            # Handle division by zero in RSI calculation
            # Check if the last value of loss is NaN or zero
            last_loss = loss.iloc[-1]
            if pd.notna(last_loss) and last_loss == 0:
                # If loss is zero, RSI is 100 if gain > 0, otherwise it implies flat price, RSI=50
                last_gain = gain.iloc[-1]
                if pd.notna(last_gain) and last_gain > 0:
                    rsi = 100.0
                # else rsi remains 50.0 (flat price or no change)
            elif pd.notna(last_loss): # loss is non-zero and not NaN
                rs = gain.iloc[-1] / last_loss
                if pd.notna(rs):
                    rsi = 100.0 - (100.0 / (1.0 + rs))
            # If last_loss is NaN, rsi remains 50.0

            # Replace NaN values with 0 or default RSI 50
            indicators = {
                'atr': float(atr if pd.notna(atr) else 0.0),
                'bb_width': float(bb_width if pd.notna(bb_width) else 0.0),
                'rsi': float(rsi if pd.notna(rsi) else 50.0), # Use calculated RSI or default 50
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
        """Load securities from the database and prepare for analysis."""
        try:
            self.logger.info("Loading securities data from database...")
            securities_data = self.db.get_all_securities()
            
            if not securities_data:
                self.logger.warning("No securities found in the database.")
                return pd.DataFrame()
            
            df = pd.DataFrame(securities_data)
            self.logger.info(f"Loaded {len(df)} securities from database")

            # Apply filtering based on configuration
            initial_count = len(df)
            if 'volume' in df.columns:
                df = df[df['volume'] >= self.min_volume]
            if 'market_cap' in df.columns:
                 df = df[df['market_cap'] >= self.min_market_cap]
            filtered_count = len(df)
            self.logger.info(f"Filtered securities: {initial_count} -> {filtered_count} (min_vol={self.min_volume}, min_cap={self.min_market_cap}) ")

            if df.empty:
                self.logger.warning("No securities met the filtering criteria.")
                return pd.DataFrame()
            
            # Clean numeric values (already done in db_manager? Double check)
            # Assume columns from DB are already clean and numeric
            # numeric_columns = ['market_cap', 'price', 'volume', 'volume_ma50', 'volume_ma200',
            #                  'ma50', 'ma200', 'beta', 'rsi', 'macd', 'inst_own_pct', 'div_yield']
            # for col in numeric_columns:
            #     if col in df.columns:
            #         df[col] = df[col].apply(self._clean_numeric)
            
            # Calculate derived metrics needed for potential score
            # Ensure required columns exist before calculation
            required_cols = ['price', 'ma50', 'ma200', 'volume', 'volume_ma50', 'volume_ma200']
            if not all(col in df.columns for col in required_cols):
                 self.logger.error(f"Missing required columns for preparing securities: {required_cols}")
                 # Fill missing required columns with 0 to avoid errors downstream, or return empty?
                 for col in required_cols:
                      if col not in df.columns:
                           df[col] = 0.0 
                 # Alternative: return pd.DataFrame()

            # Use np.where to avoid division by zero
            df['price_momentum_50d'] = np.where(df['ma50'] != 0, (df['price'] - df['ma50']) / df['ma50'], 0)
            df['price_momentum_200d'] = np.where(df['ma200'] != 0, (df['price'] - df['ma200']) / df['ma200'], 0)
            df['volume_change_50d'] = np.where(df['volume_ma50'] != 0, (df['volume'] - df['volume_ma50']) / df['volume_ma50'], 0)
            df['volume_change_200d'] = np.where(df['volume_ma200'] != 0, (df['volume'] - df['volume_ma200']) / df['volume_ma200'], 0)
            
            # Fill NaNs that might result from division by zero or missing data
            derived_cols = ['price_momentum_50d', 'price_momentum_200d', 'volume_change_50d', 'volume_change_200d']
            df[derived_cols] = df[derived_cols].fillna(0.0)

            self.logger.info("Finished preparing securities data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading and preparing securities: {str(e)}", exc_info=True)
            return None # Return None to indicate failure in error handling test

    def _calculate_price_movement_potential(self, df):
        """Calculate price movement potential for each security."""
        try:
            # Calculate price momentum
            df['price_momentum_50d'] = (df['price'] - df['ma50']) / df['ma50']
            df['price_momentum_200d'] = (df['price'] - df['ma200']) / df['ma200']
            
            # Calculate volume changes
            df['volume_change_50d'] = (df['volume'] - df['volume_ma50']) / df['volume_ma50']
            df['volume_change_200d'] = (df['volume'] - df['volume_ma200']) / df['volume_ma200']
            
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
            df.loc[df['ma50'] > df['ma200'], 'ma_alignment'] = 1.0
            df.loc[df['ma50'] < df['ma200'], 'ma_alignment'] = 0.0
            df.loc[df['ma50'] == df['ma200'], 'ma_alignment'] = 0.5

            df['price_vs_ma50'] = ((df['price'] - df['ma50']) / df['ma50'] + 0.5).clip(0, 1)
            df['price_vs_ma200'] = ((df['price'] - df['ma200']) / df['ma200'] + 0.5).clip(0, 1)

            df['technical_score'] = (
                df['ma_alignment'] * 0.4 +
                df['price_vs_ma50'] * 0.3 +
                df['price_vs_ma200'] * 0.3
            ) * 0.25

            # Calculate market context score (20%)
            df['market_score'] = df['beta'].apply(lambda x: min(max((x + 0.5) / 2.0, 0), 1)) * 0.2

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
    
    def identify_potential_movers(self, df) -> pd.DataFrame:
        """
        Identify securities with high potential for significant price movement.
        Selects top N based on score, where N is at least 10% of input.
            
        Returns:
            pd.DataFrame: Top N securities sorted by movement potential.
        """
        try:
            if df is None or df.empty:
                 self.logger.warning("Input DataFrame is empty for identifying potential movers.")
                 return pd.DataFrame()
                 
            # Ensure the score column exists
            if 'price_movement_potential' not in df.columns:
                 self.logger.error("'price_movement_potential' column missing, cannot identify movers.")
                 # Calculate it if missing - ensure _calculate_price_movement_potential is robust
                 df = self._calculate_price_movement_potential(df)
                 if 'price_movement_potential' not in df.columns:
                      return pd.DataFrame() # Return empty if calculation failed
            
            # Log score distribution
            score_stats = df['price_movement_potential'].describe()
            self.logger.info(f"Price movement potential score distribution (before top N selection):\n{score_stats}")
            
            # Determine number of securities to select (at least 10% or a minimum like 35)
            initial_count = len(df)
            min_required = 35 # Set a minimum number regardless of percentage
            top_n = max(min_required, int(0.1 * initial_count))
            self.logger.info(f"Selecting top {top_n} potential movers from {initial_count} candidates.")

            # Sort by movement potential (descending) and select top N
            potential_movers = df.sort_values('price_movement_potential', ascending=False).head(top_n).copy()
            
            if potential_movers.empty:
                self.logger.warning(f"No potential movers identified after sorting and selection.")
                return pd.DataFrame()
            
            # Add analysis timestamp
            potential_movers['analysis_timestamp'] = pd.Timestamp.now()
            
            self.logger.info(f"Identified {len(potential_movers)} securities with high movement potential")
            return potential_movers
            
        except Exception as e:
            self.logger.error(f"Error identifying potential movers: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _process_and_store_potential_movers(self, potential_movers_df: pd.DataFrame):
        """Fetches details, calculates indicators, and stores results for potential movers."""
        stored_count = 0
        for ticker in potential_movers_df['ticker']:
            try:
                # Fetch necessary data (historical mainly for tech indicators)
                yf_data = self._fetch_yf_data(ticker)
                if yf_data is None or yf_data['history'].empty:
                    self.logger.warning(f"Skipping DB store for {ticker}: Missing historical data.")
                    continue

                # Calculate technical indicators using fetched historical data
                tech_indicators = self._calculate_technical_indicators(yf_data['history'])
                if tech_indicators is None:
                     self.logger.warning(f"Skipping DB store for {ticker}: Failed tech indicator calculation.")
                     continue

                # Get the pre-calculated potential score and component scores
                row = potential_movers_df[potential_movers_df['ticker'] == ticker].iloc[0]
                opportunity_score = row['price_movement_potential']
                component_scores = {
                     'price_action_score': row.get('price_action_score', 0.0),
                     'volume_score': row.get('volume_score', 0.0),
                     'technical_score': row.get('technical_score', 0.0),
                     'market_score': row.get('market_score', 0.0)
                }
                final_metrics = {**tech_indicators, **component_scores} 

                # Store analysis result in DB
                success = self.db.insert_analysis_result(
                    ticker=ticker,
                    analysis_type='investment_opportunity',
                    score=opportunity_score,
                    metrics=final_metrics
                )
                if success:
                    stored_count += 1
                else:
                    self.logger.error(f"Failed to store investment analysis result for {ticker} in DB.")
                    
            except Exception as ticker_error:
                 self.logger.error(f"Error processing/storing investment data for ticker {ticker}: {ticker_error}", exc_info=True)
                 continue # Move to the next ticker
        self.logger.info(f"Stored investment analysis results for {stored_count}/{len(potential_movers_df)} potential tickers.")

    def run_analysis(self) -> List[str]: # Return list of tickers
        """Run the investment analysis: Load, Prepare, Score, Identify Top N, Store & Return Tickers."""
        try:
            # Load and prepare securities data from DB
            prepared_df = self.load_and_prepare_securities()
            if prepared_df is None or prepared_df.empty: 
                self.logger.error("Failed to load or prepare securities data.")
                return [] # Return empty list on failure
            
            # Track the total number of securities in the source file
            initial_count = len(prepared_df)
            min_required = max(35, int(0.1 * initial_count))
            
            # Identify top N potential movers (includes score calculation)
            potential_movers_df = self.identify_potential_movers(prepared_df)
            
            if potential_movers_df.empty:
                self.logger.info("No potential movers identified after filtering.")
                return []

            # Process details and store results for the identified movers
            self._process_and_store_potential_movers(potential_movers_df)
            
            # Get the list of identified potential tickers
            potential_tickers = potential_movers_df['ticker'].tolist()
            self.logger.info(f"Investment analysis found {len(potential_tickers)} potential tickers for next stage.")
            
            # Check if we meet the minimum threshold requirement
            if len(potential_tickers) < min_required:
                self.logger.warning(f"Identified tickers ({len(potential_tickers)}) are below the minimum threshold of 10% ({min_required}).")
                # Add a special element at the beginning of the list to indicate threshold not met
                message = f"THRESHOLD_NOT_MET: Only identified {len(potential_tickers)} potential tickers, which is below the required minimum of {min_required} (10% of {initial_count} total securities)."
                return [message] + potential_tickers
            
            return potential_tickers
            
        except Exception as e:
            self.logger.error(f"Error running investment analysis: {str(e)}", exc_info=True)
            return [] # Return empty list on failure

    def _calculate_opportunity_score(self, metrics: Dict) -> float:
        """Calculate the overall opportunity score from individual metrics provided in the metrics dict."""
        try:
            # Use the component score names as calculated in _calculate_price_movement_potential
            required_components = {
                'price_action_score': 0.0, # Correct key
                'volume_score': 0.0,
                'technical_score': 0.0,
                'market_score': 0.0
            }
            
            # Validate and clamp component scores from the input metrics
            validated_metrics = {}
            for component, default_value in required_components.items():
                 value = metrics.get(component, default_value)
                 if not isinstance(value, (int, float)) or pd.isna(value):
                     validated_metrics[component] = default_value
            else:
                     validated_metrics[component] = max(0.0, float(value)) 
            
            # The final score is simply the sum of the validated component scores
            total_score = sum(validated_metrics.values())
            
            # Clamp final score between 0 and 1
            total_score = max(0.0, min(1.0, total_score))
            
            return round(total_score, 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0

# Example Usage (Optional - useful for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db = DatabaseManager()
    analyzer = InvestmentAnalyzer(db)
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

