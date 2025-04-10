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
# from src.portfolio_manager.portfolio_manager import PortfolioManager # Removed for now

from src.utils.common import setup_logging
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
            
        logger.info("Environment setup complete.")
        return True
    except Exception as e:
        logging.error(f"Error setting up environment: {str(e)}")
        return False

def run_analysis_cycle() -> None:
    """Run a complete analysis cycle."""
    try:
        # Get logger
        logger = setup_logging('main')
        if not logger:
            return
            
        # Initialize analyzers
        investment_analyzer = InvestmentAnalyzer()
        research_analyzer = ResearchAnalyzer()
        trade_analyzer = TradeAnalyzer()
        
        # Run investment analysis with retry
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                investment_results = investment_analyzer.run_analysis()
                if not investment_results:
                    logger.warning("No securities passed investment analysis")
                    return
                break  # Success, exit retry loop
            except Exception as e:
                if "Rate limit" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Rate limit exceeded, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise  # Re-raise if not a rate limit error or last attempt
            
        # Run research analysis
        research_results = research_analyzer.run_risk_analysis()
        if not research_results:
            logger.warning("No securities passed research analysis")
            return
            
        # Run trade analysis
        trade_results = trade_analyzer.generate_trade_signals()
        if not trade_results:
            logger.warning("No trade recommendations generated")
            return
            
        logger.info("Analysis cycle completed successfully")
    except Exception as e:
        logger.error(f"Error in analysis cycle: {str(e)}")

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