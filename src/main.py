import logging
from datetime import datetime
import json
from pathlib import Path
import os
import time
import schedule
from dotenv import load_dotenv
import pandas as pd
import traceback

# Import the refactored analyzer classes
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.trade_analyst.trade_analyzer import TradeAnalyzer
from src.database import DatabaseManager

# Import common utility functions
from src.utils.common import setup_logging, save_recommendations_to_csv
# from src.utils.json_utils import load_from_json # No longer needed here
from src.utils.logger import get_logger
from src.utils.config import setup_environment

# Constants
MAX_RISK_THRESHOLD = 80  # Maximum acceptable risk score for trade analysis
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

def save_to_json(data, filepath):
    """Save data to a JSON file."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def setup_environment() -> bool:
    """Set up the environment for the application."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Set up logging
        logger = setup_logging('main')
        if not logger:
            return False
            
        # Initialize database
        db = DatabaseManager()
        logger.info("Database initialized successfully")
            
        logger.info("Environment setup complete.")
        return True
    except Exception as e:
        logging.error(f"Error setting up environment: {str(e)}")
        return False

def run_analysis_cycle() -> None:
    """Run a complete analysis cycle."""
    logger = setup_logging('main')
    if not logger:
        return

    try:
        # Initialize database and analyzers
        db = DatabaseManager()
        investment_analyzer = InvestmentAnalyzer(db)
        research_analyzer = ResearchAnalyzer(db)
        trade_analyzer = TradeAnalyzer(db)
        
        # 1. Investment Analysis: Identify Top N tickers
        logger.info("Starting Investment Analysis...")
        potential_tickers = investment_analyzer.run_analysis()
        if not potential_tickers:
            logger.warning("Investment analysis did not identify any potential tickers.")
            return
            
        # Check if the first element is a threshold not met message
        threshold_message = None
        if potential_tickers and isinstance(potential_tickers[0], str) and potential_tickers[0].startswith("THRESHOLD_NOT_MET:"):
            threshold_message = potential_tickers[0]
            # Remove the message from the list
            potential_tickers = potential_tickers[1:]
            logger.warning(threshold_message)
            
        logger.info(f"Investment Analysis identified {len(potential_tickers)} potential tickers.")

        # 2. Research Analysis: Calculate risk for identified tickers
        logger.info("Starting Research Analysis...")
        
        # If there was a threshold message, notify the Research Analyzer
        if threshold_message:
            logger.info("Passing threshold warning message to Research Analyzer.")
            # We could pass this message to the Research Analyzer if needed
            # For now, just log it and continue with the available tickers
        
        risk_results = research_analyzer.run_risk_analysis(potential_tickers)
        if not risk_results:
            logger.warning("Research analysis failed to produce risk results for identified tickers.")
            # Continue to trade analysis? Or stop? Let's continue but trade analyzer will skip.
            # return # Stop if research fails
        processed_risk_tickers = list(risk_results.keys())
        logger.info(f"Research Analysis processed {len(processed_risk_tickers)} tickers.")
            
        # 3. Trade Analysis: Generate signals for tickers that passed investment and research
        logger.info("Starting Trade Analysis...")
        # Pass the list of tickers for which risk analysis was successfully completed
        trade_signals = trade_analyzer.generate_trade_signals(processed_risk_tickers)
        if not trade_signals:
            logger.warning("Trade analysis generated no recommendations.")
            return
        logger.info(f"Trade Analysis generated {len(trade_signals)} recommendations.")
            
        # 4. Save Trade Recommendations to CSV
        # Convert trade_signals dict to list of dicts for CSV saving, filtering out None values
        recommendations = []
        for ticker, data in trade_signals.items():
            if data is None: # Skip if signal generation failed for this ticker
                 continue 
                 
            security_info = db.get_security(ticker) # Get name
            recommendations.append({
                'ticker': ticker,
                'name': security_info.get('name', 'N/A') if security_info else 'N/A',
                'signal': data.get('signal', 'HOLD'), # Use 'signal' key now
                'timeframe': data.get('timeframe', 'N/A'), 
                'confidence': data.get('confidence', 'NEUTRAL'), 
                'price': data.get('current_price', 0.0),
                'position_size': data.get('position_size', 0.0),
                # Justification is already JSON stringified in _generate_signals output dict
                'justification': data.get('justification', '[]') 
            })
            
        if recommendations:
             output_path = os.path.join(OUTPUT_DIR, 'Trade_Recommendations_latest.csv')
             success = save_recommendations_to_csv(recommendations, output_path)
             if success:
                  logger.info(f"Successfully saved {len(recommendations)} recommendations to {output_path}")
             else:
                  logger.error(f"Failed to save recommendations to {output_path}")
        else:
             logger.info("No recommendations formatted for CSV output.")

        logger.info("Analysis cycle completed successfully")
    except Exception as e:
        logger.error(f"Error in analysis cycle: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to run the AI Trader framework."""
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not setup_environment():
        logging.critical("Environment setup failed. Exiting.")
        return
        
    logger = get_logger('main')
    logger.info("Starting AI Trader framework")
    
    # Run a single analysis cycle
    logger.info("Running a single analysis cycle.")
    run_analysis_cycle()
    logger.info("AI Trader framework finished.")

if __name__ == "__main__":
    main()