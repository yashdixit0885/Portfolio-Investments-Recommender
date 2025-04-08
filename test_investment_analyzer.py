import os
import sys
import logging
from datetime import datetime
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.utils.common import setup_logging, get_current_time, load_from_json

def setup_test_environment():
    """Set up the test environment with necessary directories and logging."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Set up logging
    logger = setup_logging('test_investment_analyzer')
    logger.info("Starting investment analysis test")
    
    return logger

def test_investment_analyzer(data_file: str = None, logger: logging.Logger = None):
    """Test the InvestmentAnalyzer with sample or real data."""
    try:
        # Initialize analyzers
        investment_analyzer = InvestmentAnalyzer()
        research_analyzer = ResearchAnalyzer()
        
        # Process securities data
        if data_file and os.path.exists(data_file):
            logger.info(f"Processing securities data from {data_file}")
            opportunities = investment_analyzer.process_securities_data(data_file)
        else:
            logger.info("No data file provided or file not found")
            return
            
        if not opportunities:
            logger.info("No investment opportunities found")
            return
            
        # Generate trade recommendations from investment analyzer
        trade_recommendations = investment_analyzer.generate_trade_recommendations(opportunities)
        
        if not trade_recommendations:
            logger.info("No trade recommendations generated")
            return
            
        # Process opportunities with research analyzer
        logger.info("\nProcessing opportunities with Research Analyzer...")
        research_analyzer.process_flagged_opportunities(opportunities)
        
        # Load the research analysis
        research_analysis = load_from_json(research_analyzer.analysis_file)
        
        if not research_analysis:
            logger.info("No research analysis generated")
            return
            
        # Display combined analysis
        logger.info("\nCombined Analysis Results:")
        logger.info("=" * 80)
        
        for rec in trade_recommendations:
            ticker = rec['ticker']
            logger.info(f"\nTicker: {ticker}")
            logger.info("-" * 40)
            
            # Get investment analyzer recommendation
            logger.info("Investment Analyzer Recommendation:")
            logger.info(f"Action: {rec['action']}")
            logger.info(f"Score: {rec['score']:.4f}")
            logger.info(f"Position Size: {rec['position_size']}")
            logger.info(f"Rationale: {rec['rationale']}")
            
            # Get research analyzer analysis
            research_rec = next((r for r in research_analysis if r['ticker'] == ticker), None)
            
            if research_rec:
                logger.info("\nResearch Analyzer Analysis:")
                logger.info(f"Action: {research_rec['action']}")
                logger.info(f"Confidence Score: {research_rec['confidence_score']:.4f}")
                logger.info(f"Time Horizon: {research_rec['time_horizon']}")
                logger.info(f"Position Size: {research_rec['position_size']}")
                logger.info(f"Risk Parameters: {research_rec['risk_parameters']}")
                logger.info(f"Rationale: {research_rec['rationale']}")
            
            logger.info("=" * 80)
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up test environment
    logger = setup_test_environment()
    
    # Get data file from command line argument or use default
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'data/securities_data.csv'
    
    # Run test
    test_investment_analyzer(data_file, logger) 