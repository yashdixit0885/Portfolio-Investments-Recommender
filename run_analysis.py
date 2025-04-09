import logging
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.trade_analyst.trade_analyzer import TradeAnalyzer
from src.utils.common import load_from_json
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def display_results(file_path, title):
    data = load_from_json(file_path)
    if not data:
        print(f"\nNo data found in {file_path}")
        return
    
    if "risk_scored_securities" in data:
        securities = data["risk_scored_securities"]
    elif "high_potential_securities" in data:
        securities = data["high_potential_securities"]
    else:
        securities = data
    
    print(f"\n{title}")
    print("=" * 80)
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(securities)
    
    # Select and rename columns for display
    display_cols = {
        'Symbol': 'Symbol',
        'price_movement_potential': 'Movement Potential',
        'risk_score': 'Risk Score',
        'Signal': 'Signal',
        'Timeframe': 'Timeframe',
        'Current Price': 'Price',
        'Market Cap': 'Market Cap',
        'Volume': 'Volume'
    }
    
    # Filter columns that exist in the DataFrame
    display_cols = {k: v for k, v in display_cols.items() if k in df.columns}
    
    # Display the results
    print(df[list(display_cols.keys())].head(10).to_string())
    print(f"\nTotal securities analyzed: {len(securities)}")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Run Investment Analyzer
        logger.info("Running Investment Analyzer...")
        investment_analyzer = InvestmentAnalyzer()
        investment_analyzer.run_analysis()
        display_results('data/high_potential_securities.json', "High Potential Securities")
        
        # Run Research Analyzer
        logger.info("Running Research Analyzer...")
        research_analyzer = ResearchAnalyzer()
        research_analyzer.run_risk_analysis()
        display_results('data/risk_scored_securities.json', "Risk Scored Securities")
        
        # Run Trade Analyzer
        logger.info("Running Trade Analyzer...")
        trade_analyzer = TradeAnalyzer()
        trade_analyzer.generate_trade_signals()
        
        # Display trade recommendations from CSV
        try:
            trade_recs = pd.read_csv('output/Trade_Recommendations_latest.csv')
            print("\nTrade Recommendations")
            print("=" * 80)
            print(trade_recs[['Ticker', 'Signal', 'Timeframe', 'Risk Score', 'Current Price', 'Justification']].head(10).to_string())
            print(f"\nTotal trade recommendations: {len(trade_recs)}")
        except Exception as e:
            logger.error(f"Error reading trade recommendations: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}")

if __name__ == "__main__":
    main() 