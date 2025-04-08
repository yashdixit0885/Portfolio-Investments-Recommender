import os
import time
import schedule
import logging
from datetime import datetime
from dotenv import load_dotenv
from investment_analyst.investment_analyzer import InvestmentAnalyzer
from research_analyst.research_analyzer import ResearchAnalyzer
from trade_analyst.trade_analyzer import TradeAnalyzer
from portfolio_manager.portfolio_manager import PortfolioManager
from utils.common import setup_logging

def setup_environment():
    """Setup environment variables and logging"""
    load_dotenv()
    logger = setup_logging('main')
    
    # Verify required environment variables
    required_vars = ['SECURITIES_DATA_PATH']  # Path to securities data spreadsheet
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
        
    return True

def run_analysis_cycle():
    """Run one complete analysis cycle"""
    logger = logging.getLogger('main')
    logger.info("Starting analysis cycle")
    
    try:
        # 1. Investment Analysis
        investment_analyzer = InvestmentAnalyzer()
        securities_file = os.getenv('SECURITIES_DATA_PATH')
        if not investment_analyzer.process_securities_file(securities_file):
            logger.error("Failed to process securities data")
            return
        
        # 2. Research Analysis
        research_analyzer = ResearchAnalyzer()
        research_analyzer.process_flagged_opportunities()
        
        # 3. Trade Analysis
        trade_analyzer = TradeAnalyzer()
        trade_analyzer.process_recommendations()
        
        # 4. Portfolio Management
        portfolio_manager = PortfolioManager()
        portfolio_manager.process_trade_analysis()
        
        logger.info("Analysis cycle completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analysis cycle: {str(e)}")

def main():
    """Main function to run the AI Trader framework"""
    if not setup_environment():
        return
        
    logger = logging.getLogger('main')
    logger.info("Starting AI Trader framework")
    
    # Schedule analysis cycles
    schedule.every(5).minutes.do(run_analysis_cycle)
    
    # Run initial cycle
    run_analysis_cycle()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()