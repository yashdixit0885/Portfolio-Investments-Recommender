import os
import time
import schedule
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import the refactored analyzer classes
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.trade_analyst.trade_analyzer import TradeAnalyzer
# from src.portfolio_manager.portfolio_manager import PortfolioManager # Removed for now

from src.utils.common import setup_logging
# from src.utils.json_utils import load_from_json # No longer needed here

def setup_environment():
    """Load .env file (if exists) and setup logging"""
    load_dotenv() # Load environment variables from .env file, if present
    logger = setup_logging('main') # Setup centralized logging
    logger.info("Environment setup complete.")
    return True

def run_analysis_cycle():
    """Run one complete analysis cycle from Investment to Trade Analysis."""
    logger = logging.getLogger('main')
    logger.info("Starting new analysis cycle...")
    
    cycle_successful = True
    
    try:
        # 1. Investment Analysis
        logger.info("[Stage 1/3] Running Investment Analysis...")
        investment_analyzer = InvestmentAnalyzer()
        if not investment_analyzer.run_analysis():
            logger.error("Investment Analysis failed or produced no results. Stopping cycle.")
            cycle_successful = False
            return # Stop the cycle if this stage fails
        logger.info("Investment Analysis completed.")
        
        # 2. Research Analysis
        logger.info("[Stage 2/3] Running Research Analysis (Risk Scoring)...")
        research_analyzer = ResearchAnalyzer()
        if not research_analyzer.run_risk_analysis():
            logger.error("Research Analysis failed or produced no results. Stopping cycle.")
            cycle_successful = False
            return # Stop the cycle if this stage fails
        logger.info("Research Analysis completed.")
        
        # 3. Trade Analysis
        logger.info("[Stage 3/3] Running Trade Analysis (Signal Generation)...")
        trade_analyzer = TradeAnalyzer()
        if not trade_analyzer.generate_trade_signals():
            logger.error("Trade Analysis failed or produced no results.")
            cycle_successful = False
            # Allow cycle to finish logging even if last step fails
        else:
            logger.info("Trade Analysis completed.")
            
        # 4. Portfolio Management (Removed for now)
        # logger.info("[Stage 4/4] Running Portfolio Management...")
        # portfolio_manager = PortfolioManager()
        # portfolio_manager.process_trade_analysis()
        # logger.info("Portfolio Management completed.")
        
        if cycle_successful:
            logger.info("Analysis cycle completed successfully.")
        else:
             logger.warning("Analysis cycle completed with errors.")
        
    except Exception as e:
        logger.critical(f"Critical error during analysis cycle: {str(e)}", exc_info=True)

def main():
    """Main function to run the AI Trader framework."""
    if not setup_environment():
        logging.critical("Environment setup failed. Exiting.")
        return
        
    logger = logging.getLogger('main')
    logger.info("Starting AI Trader framework")
    
    # --- Scheduling Option (Commented out for single run) ---
    # logger.info("Scheduling analysis cycle every 5 minutes.")
    # schedule.every(5).minutes.do(run_analysis_cycle)
    # run_analysis_cycle() # Run once immediately
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60) # Check every minute
    # --- End Scheduling Option ---
    
    # --- Single Run Option ---
    logger.info("Running a single analysis cycle.")
    run_analysis_cycle()
    logger.info("AI Trader framework finished.")
    # --- End Single Run Option ---

if __name__ == "__main__":
    main()