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

def run_analysis_cycle():
    """Run a complete analysis cycle."""
    logger = get_logger()
    logger.info("Starting new analysis cycle")
    
    try:
        # Initialize analyzers
        research_analyzer = ResearchAnalyzer()
        trade_analyzer = TradeAnalyzer()
        
        # Read securities from CSV file
        securities_file = os.path.join(DATA_DIR, 'securities_data.csv')
        if not os.path.exists(securities_file):
            logger.error(f"Securities data file not found: {securities_file}")
            return
            
        # Read securities from CSV using pandas
        try:
            securities_df = pd.read_csv(securities_file)
            potential_securities = securities_df['Symbol'].dropna().unique().tolist()
            logger.info(f"Found {len(potential_securities)} unique securities for analysis")
        except Exception as e:
            logger.error(f"Error reading securities data: {str(e)}")
            logger.error(traceback.format_exc())
            return
        
        if not potential_securities:
            logger.warning("No potential securities found for analysis")
            return
            
        # Run research analysis
        try:
            analysis_results = research_analyzer.run_risk_analysis(potential_securities)
            if not analysis_results:
                logger.warning("No securities passed research analysis")
                return
            logger.info(f"Research analysis completed for {len(analysis_results)} securities")
        except Exception as e:
            logger.error(f"Error in research analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return
            
        # Process each analyzed security through trade analysis
        trade_recommendations = []
        for symbol, result in analysis_results.items():
            try:
                if result['risk_score'] > MAX_RISK_THRESHOLD:
                    logger.debug(f"Skipping {symbol} due to high risk score: {result['risk_score']}")
                    continue
                    
                historical_data = result['historical_data']
                trade_analysis = trade_analyzer.analyze_security(historical_data)
                
                if trade_analysis:
                    trade_recommendations.append({
                        'symbol': symbol,
                        'risk_score': result['risk_score'],
                        'trade_score': trade_analysis.get('trade_score', 0),
                        'recommendation': trade_analysis.get('recommendation', 'HOLD'),
                        'justification': trade_analysis.get('justification', '')
                    })
                    logger.info(f"Added trade recommendation for {symbol}")
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Save final recommendations
        if trade_recommendations:
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'recommendations': trade_recommendations
            }
            output_file = os.path.join(OUTPUT_DIR, 'trade_recommendations.json')
            save_to_json(output_data, output_file)
            logger.info(f"Generated {len(trade_recommendations)} trade recommendations")
        else:
            logger.warning("No trade recommendations generated")
            
    except Exception as e:
        logger.error(f"Error in analysis cycle: {str(e)}")
        logger.error(traceback.format_exc())

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