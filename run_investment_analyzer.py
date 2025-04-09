import logging
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
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
    
    if "high_potential_securities" in data:
        securities = data["high_potential_securities"]
    else:
        securities = data
    
    print(f"\n{title}")
    print("=" * 80)
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(securities)
    
    # Display all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Display the results
    print(df.head(10).to_string())
    print(f"\nTotal securities analyzed: {len(securities)}")
    
    # Display column names for reference
    print("\nColumns in the output:")
    for col in df.columns:
        print(f"- {col}")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Run Investment Analyzer
        logger.info("Running Investment Analyzer...")
        investment_analyzer = InvestmentAnalyzer()
        investment_analyzer.run_analysis()
        
        # Display results
        display_results('data/high_potential_securities.json', "High Potential Securities")
        
    except Exception as e:
        logger.error(f"Error running Investment Analyzer: {str(e)}")

if __name__ == "__main__":
    main() 