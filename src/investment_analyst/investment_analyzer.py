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
        
    def _clean_numeric(self, x):
        """Cleans and converts input to a float, handling N/A, percentages, and commas."""
        if pd.isna(x) or x == 'N/A':
            return np.nan # Use NaN for missing values initially
        if isinstance(x, (int, float)):
            return float(x)
        # Remove any spaces and handle percentage signs
        x = str(x).strip().replace(' ', '')
        # Remove percentage sign and handle plus signs
        x = x.replace('%', '').replace('+', '')
        # Remove commas from numbers
        x = x.replace(',', '')
        try:
            return float(x)
        except ValueError:
            self.logger.debug(f"Could not convert '{x}' to float, returning NaN")
            return np.nan

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

    def _calculate_price_movement_potential(self, row: pd.Series) -> float:
        """Calculate a score indicating the potential for significant price movement (magnitude)."""
        try:
            # Normalize and weight factors contributing to potential movement
            
            # 1. Volatility/Beta (25%): Higher beta suggests higher potential volatility
            beta = row.get('Beta', 1.0) # Default to 1.0 if missing
            beta_score = min(1.0, max(0, (beta / 2.0))) # Normalize roughly (e.g., beta 2.0 = max score)
            
            # 2. Price Momentum (Absolute) (35%): Larger distance from MAs indicates potential
            abs_momentum_50d = abs(row.get('price_momentum_50d', 0))
            abs_momentum_200d = abs(row.get('price_momentum_200d', 0))
            # Normalize (e.g., 20% move = max score)
            norm_momentum_50d = min(1.0, abs_momentum_50d / 0.20) 
            norm_momentum_200d = min(1.0, abs_momentum_200d / 0.20)
            momentum_score = (norm_momentum_50d * 0.6) + (norm_momentum_200d * 0.4)
            
            # 3. Volume Surge (Positive) (25%): High volume increases confidence in moves
            vol_change_50d = max(0, row.get('volume_change_50d', 0)) # Only consider positive change
            vol_change_200d = max(0, row.get('volume_change_200d', 0))
            # Normalize (e.g., 2x average volume = max score)
            norm_vol_50d = min(1.0, vol_change_50d / 1.0) # 1.0 = 100% increase = 2x avg vol
            norm_vol_200d = min(1.0, vol_change_200d / 1.0)
            volume_score = (norm_vol_50d * 0.6) + (norm_vol_200d * 0.4)
            
            # 4. RSI Extremes (15%): Further from 50 indicates more potential for reversion/trend
            rsi = row.get('rsi', 0.5) * 100 # Assuming RSI was stored as 0-1
            rsi_distance = abs(rsi - 50)
            rsi_score = min(1.0, rsi_distance / 30) # Normalize (e.g., RSI 20 or 80 = max score)

            # Calculate final score
            final_score = (
                beta_score * 0.25 +
                momentum_score * 0.35 +
                volume_score * 0.25 +
                rsi_score * 0.15
            )
            
            return round(final_score, 4)
            
        except Exception as e:
            self.logger.warning(f"Error calculating movement potential for {row.get('Symbol', 'Unknown')}: {str(e)}")
            return 0.0
    
    def identify_potential_movers(self) -> List[Dict]:
        """Loads data, calculates movement potential, selects top 10%, and returns them."""
        df = self.load_and_prepare_securities()
        
        if df.empty:
            self.logger.error("Failed to load or prepare securities data. Cannot identify potential movers.")
            return []
            
        # Calculate potential score for each security
        df['movement_potential_score'] = df.apply(self._calculate_price_movement_potential, axis=1)
        
        # Sort by score
        df_sorted = df.sort_values(by='movement_potential_score', ascending=False)
        
        # Determine number to select (at least 10%)
        total_securities = len(df_sorted)
        num_to_select = max(1, int(total_securities * 0.10)) 
        
        self.logger.info(f"Calculated movement potential for {total_securities} securities.")
        self.logger.info(f"Selecting top {num_to_select} securities (>= 10%) for further analysis.")
        
        # Select top N
        top_securities_df = df_sorted.head(num_to_select)
        
        # Convert to list of dictionaries for saving/passing on
        # Select relevant columns to pass to the Research Analyst
        columns_to_keep = [
            'Symbol', 'Name', 'Last', 'Industry', 'Sector', 'Market Cap', 'Volume',
            'price_momentum_50d', 'price_momentum_200d', 
            'volume_change_50d', 'volume_change_200d',
            'rsi', 'Beta', 'movement_potential_score'
        ]
        # Filter columns that actually exist in the dataframe
        existing_columns_to_keep = [col for col in columns_to_keep if col in top_securities_df.columns]
        
        potential_movers = top_securities_df[existing_columns_to_keep].to_dict('records')
        
        # Add timestamp
        timestamp = get_current_time().isoformat()
        for mover in potential_movers:
            mover['identified_at'] = timestamp
            # Convert NaN to None for JSON compatibility
            for key, value in mover.items():
                 if pd.isna(value):
                     mover[key] = None

        return potential_movers

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
        
        if not potential_movers:
             self.logger.warning("Investment Analysis did not identify any securities.")
             return False
             
        success = self.save_potential_securities(potential_movers)
        
        if success:
            self.logger.info("Investment Analyzer finished successfully.")
        else:
            self.logger.error("Investment Analyzer finished with errors during saving.")
            
        return success

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

