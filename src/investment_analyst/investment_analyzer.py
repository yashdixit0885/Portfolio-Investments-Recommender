"""
Identifies securities with high potential for significant price movement.

The core logic resides in the `_calculate_price_movement_potential` method. 
It calculates a score based on several factors, weighted to prioritize indicators 
of potential volatility and significant price changes, regardless of direction:

1. Volatility/Beta (25%): Higher beta suggests higher potential for larger price swings.
   Normalized based on a beta of 2.0 representing maximum contribution.

2. Absolute Price Momentum (35%): The magnitude of the price's deviation from its 
   50-day and 200-day moving averages. Larger deviations suggest the price is 
   already moving significantly or is far from its recent trends.
   Normalized based on a 20% deviation representing maximum contribution.

3. Volume Surge (25%): Significant increases in volume compared to the 50-day 
   and 200-day averages. High volume can confirm the strength behind a price move.
   Normalized based on volume doubling (100% increase) representing maximum contribution.

4. RSI Extremes (15%): How far the Relative Strength Index (RSI) is from the neutral 
   midpoint of 50. Values closer to 0 or 100 indicate potentially overbought or 
   oversold conditions, which can precede significant price reversals or continuations.
   Normalized based on RSI reaching 20 or 80 representing maximum contribution.

The overall score aggregates these factors to rank securities by their likelihood 
of experiencing substantial price movement, making them candidates for further 
detailed research.
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from src.utils.common import save_to_json, load_from_json, get_current_time, save_recommendations_to_csv
import numpy as np

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
        
        # Load configuration (though less critical now, kept for potential future use)
        self.load_config()
        
    def load_config(self):
        """Load configuration from environment variables or use defaults."""
        # These thresholds are less relevant for the new goal but kept for context
        self.min_volume = int(os.getenv('MIN_VOLUME', '500000')) 
        self.min_market_cap = float(os.getenv('MIN_MARKET_CAP', '50000000'))
        
    def _clean_numeric(self, value):
        """Clean and convert numeric values from strings to floats."""
        if pd.isna(value) or value is None or value == '':
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        # Remove any non-numeric characters except decimal point and minus sign
        cleaned = ''.join(c for c in str(value) if c.isdigit() or c in '.-')
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    def load_and_prepare_securities(self) -> pd.DataFrame:
        """Loads securities data from the CSV file, cleans it, and calculates necessary metrics."""
        try:
            if not os.path.exists(self.input_file):
                 self.logger.error(f"Input file not found: {self.input_file}")
                 return pd.DataFrame()

            df = pd.read_csv(self.input_file)
            self.logger.info(f"Loaded {len(df)} securities from {self.input_file}")
            self.logger.debug(f"Available columns: {df.columns.tolist()}")

            # Define columns to clean and convert
            percentage_columns = ['% Insider', 'Div Yield(a)', 'Sales %(q)', 'Sales %(a)', 'Profit%', '5Y Rev%', '14D Rel Str', '% Institutional']
            numeric_columns = ['Last', 'Volume', 'Market Cap', 'Price Vol', '200D Avg Vol', 
                             '50D Avg Vol', '14D MACD', '200D MA', '50D MA', 'P/C OI',
                             'Mean Target', 'Beta', 'Debt/Equity', 'Price/Book', 'P/E fwd']

            # Clean percentage columns
            for col in percentage_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self._clean_numeric) / 100.0 # Convert percentage right away
                else:
                    self.logger.warning(f"Percentage column '{col}' not found in input file.")

            # Clean other numeric columns
            for col in numeric_columns:
                 if col in df.columns:
                     df[col] = df[col].apply(self._clean_numeric)
                 else:
                     self.logger.warning(f"Numeric column '{col}' not found in input file.")
            
            # --- Calculate Metrics for Price Movement Potential ---
            
            # Volume change relative to averages
            df['volume_change_50d'] = ((df['Volume'] / df['50D Avg Vol']) - 1).fillna(0)
            df['volume_change_200d'] = ((df['Volume'] / df['200D Avg Vol']) - 1).fillna(0)
            
            # Price momentum relative to MAs
            df['price_momentum_50d'] = ((df['Last'] / df['50D MA']) - 1).fillna(0)
            df['price_momentum_200d'] = ((df['Last'] / df['200D MA']) - 1).fillna(0)
            
            # Rename for clarity if needed (example: RSI)
            if '14D Rel Str' in df.columns:
                df.rename(columns={'14D Rel Str': 'rsi'}, inplace=True)
            elif 'rsi' not in df.columns:
                 df['rsi'] = 0.5 # Default to neutral if missing

            # Ensure essential columns exist for scoring
            required_cols = ['Symbol', 'Name', 'Last', 'volume_change_50d', 'volume_change_200d', 
                             'price_momentum_50d', 'price_momentum_200d', 'rsi', 'Beta']
            for col in required_cols:
                if col not in df.columns:
                     self.logger.warning(f"Required column for scoring '{col}' not found. Filling with default.")
                     if col == 'Beta':
                         df[col] = 1.0 # Default Beta
                     elif col in ['Symbol', 'Name', 'Last']:
                         # These are critical, drop rows if missing fundamental identifiers
                         self.logger.error(f"Critical column '{col}' missing. Cannot proceed reliably.")
                         return pd.DataFrame() # Or handle more gracefully
                     else:
                         df[col] = 0.0 # Default other metrics to 0

            return df

        except Exception as e:
            self.logger.error(f"Error loading and preparing securities data: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _calculate_price_movement_potential(self, securities_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the price movement potential for each security based on technical indicators.
        
        Args:
            securities_data (pd.DataFrame): DataFrame containing securities data with RSI, MACD, and volume metrics.
        
        Returns:
            pd.DataFrame: DataFrame with added price_movement_potential column.
        """
        try:
            # Create a copy to avoid modifying the original DataFrame
            result_df = securities_data.copy()
            
            # Calculate RSI potential (higher RSI indicates overbought, lower indicates oversold)
            result_df['rsi_potential'] = result_df['rsi'].apply(lambda x: 
                1.0 if x <= 30 else  # Strong oversold signal
                0.8 if x <= 40 else  # Moderate oversold signal
                0.2 if x >= 70 else  # Strong overbought signal
                0.4 if x >= 60 else  # Moderate overbought signal
                0.5  # Neutral
            )
            
            # Calculate MACD potential based on MACD crossing signal line
            result_df['macd_potential'] = (
                (result_df['14D MACD'] - result_df['14D MACD_Signal']).apply(lambda x:
                    1.0 if x > 2 else  # Strong bullish signal
                    0.8 if x > 0 else  # Moderate bullish signal
                    0.2 if x < -2 else  # Strong bearish signal
                    0.4 if x < 0 else  # Moderate bearish signal
                    0.5  # Neutral
                )
            )
            
            # Calculate volume potential based on volume changes
            result_df['volume_potential'] = (
                (result_df['Volume'] / result_df['50D Avg Vol']).apply(lambda x:
                    1.0 if x > 2.0 else  # Very high volume
                    0.8 if x > 1.5 else  # High volume
                    0.6 if x > 1.2 else  # Moderate volume
                    0.4 if x > 0.8 else  # Low volume
                    0.2  # Very low volume
                )
            )
            
            # Calculate weighted average of all potentials
            weights = {
                'rsi_potential': 0.4,
                'macd_potential': 0.4,
                'volume_potential': 0.2
            }
            
            result_df['price_movement_potential'] = (
                result_df['rsi_potential'] * weights['rsi_potential'] +
                result_df['macd_potential'] * weights['macd_potential'] +
                result_df['volume_potential'] * weights['volume_potential']
            )
            
            # Drop intermediate columns
            result_df = result_df.drop(['rsi_potential', 'macd_potential', 'volume_potential'], axis=1)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating price movement potential: {str(e)}")
            # Return original DataFrame with a default price_movement_potential column
            securities_data['price_movement_potential'] = 0.5  # Neutral score
            return securities_data
    
    def identify_potential_movers(self) -> pd.DataFrame:
        """
        Identify potential movers based on price movement potential and other metrics.
        
        Returns:
            pd.DataFrame: DataFrame containing potential movers with their scores and metrics.
        """
        try:
            # Load and prepare securities data
            df = self.load_and_prepare_securities()
            
            # Calculate price movement potential for all securities
            df = self._calculate_price_movement_potential(df)
            
            # Filter for securities with high movement potential
            high_potential = df[df['price_movement_potential'] > 0.7].copy()
            
            # Sort by movement potential in descending order
            high_potential = high_potential.sort_values('price_movement_potential', ascending=False)
            
            # Add additional analysis if needed
            high_potential['analysis_timestamp'] = pd.Timestamp.now()
            
            return high_potential
            
        except Exception as e:
            self.logger.error(f"Error in identify_potential_movers: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def save_potential_securities(self, securities: List[Dict[str, Any]]) -> bool:
        """Saves the list of high-potential securities to a JSON file."""
        if not securities:
            self.logger.warning("No high-potential securities found or provided to save.")
            return False
        try:
            if save_to_json(securities, self.output_file):
                self.logger.info(f"Saved {len(securities)} high-potential securities to {self.output_file}")
                return True
            else:
                self.logger.error(f"Failed to save high-potential securities to {self.output_file}")
                return False
        except Exception as e:
            self.logger.error(f"Error saving high-potential securities: {str(e)}")
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
        """Calculate an opportunity score based on various metrics."""
        try:
            # Initialize component scores
            momentum_score = 0.0
            volume_score = 0.0
            technical_score = 0.0
            fundamental_score = 0.0
            
            # Calculate momentum score (30%)
            if 'price_momentum_50d' in metrics and 'price_momentum_200d' in metrics:
                momentum_50d = metrics['price_momentum_50d']
                momentum_200d = metrics['price_momentum_200d']
                momentum_score = (momentum_50d * 0.6 + momentum_200d * 0.4) * 0.3
            
            # Calculate volume score (20%)
            if 'volume_change_50d' in metrics and 'volume_change_200d' in metrics:
                volume_50d = metrics['volume_change_50d']
                volume_200d = metrics['volume_change_200d']
                volume_score = (volume_50d * 0.6 + volume_200d * 0.4) * 0.2
            
            # Calculate technical score (25%)
            if 'rsi' in metrics and 'macd' in metrics:
                rsi = metrics['rsi']
                macd = metrics['macd']
                technical_score = (
                    (1 - abs(rsi - 50) / 50) * 0.5 +  # RSI component
                    (1 if macd > 0 else 0) * 0.5      # MACD component
                ) * 0.25
            
            # Calculate fundamental score (25%)
            if 'pe_ratio' in metrics and 'price_to_book' in metrics:
                pe = metrics['pe_ratio']
                pb = metrics['price_to_book']
                fundamental_score = (
                    (1 / (1 + abs(pe - 15) / 15)) * 0.5 +  # PE component
                    (1 / (1 + abs(pb - 2) / 2)) * 0.5      # PB component
                ) * 0.25
            
            # Calculate final score
            final_score = momentum_score + volume_score + technical_score + fundamental_score
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, final_score))
            
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

