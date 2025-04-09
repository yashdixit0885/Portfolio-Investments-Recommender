"""
Portfolio Risk Analysis Tool

This script analyzes the risk levels of different stocks and securities in your investment portfolio.
It helps investors understand how risky each investment is on a scale from 0 (safest) to 100 (riskiest).

What the script does:
1. Reads a list of stocks from a CSV file (securities_data.csv)
2. For each stock, it calculates a risk score based on multiple factors:
   - Market volatility (how much the stock price moves up and down)
   - Company size and financial health
   - Industry sector risks
   - Trading volume and liquidity
   - Technical indicators like RSI and Beta
3. Saves the results in an easy-to-read format (JSON file)

The risk scores help investors:
- Identify which stocks are safer investments
- Understand which stocks might be more volatile
- Make informed decisions about portfolio diversification
- Balance their portfolio between high-risk and low-risk investments

Example risk score ranges:
- 0-25: Very low risk (stable, established companies)
- 26-50: Moderate risk (growing companies with some volatility)
- 51-75: High risk (volatile stocks, emerging companies)
- 76-100: Very high risk (speculative investments)

The results are saved in 'data/direct_risk_analysis.json' for further analysis.
"""

import os
import pandas as pd
import logging
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.utils.common import setup_logging, save_to_json

def run_direct_risk_analysis():
    # Setup logging
    logger = setup_logging('direct_risk_analysis')
    logger.info("Starting direct risk analysis on securities_data.csv")
    
    # Initialize the analyzer
    analyzer = ResearchAnalyzer()
    
    try:
        # Read the CSV file
        csv_path = os.path.join('src', 'data', 'securities_data.csv')
        if not os.path.exists(csv_path):
            logger.error(f"Input file not found: {csv_path}")
            return False
            
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} securities from CSV")
        
        # Convert DataFrame rows to list of dictionaries
        securities = df.to_dict('records')
        
        # Process each security
        risk_scored_results = []
        for security in securities:
            # Rename Symbol to Ticker if needed
            if 'Symbol' in security and 'Ticker' not in security:
                security['Ticker'] = security['Symbol']
                
            try:
                risk_score = analyzer._calculate_risk_score(security)
                security['risk_score'] = risk_score
                risk_scored_results.append(security)
                logger.info(f"Calculated risk score for {security['Ticker']}: {risk_score}")
            except Exception as e:
                logger.error(f"Failed to calculate risk score for {security.get('Ticker', 'Unknown')}: {str(e)}")
                
        # Sort by risk score
        risk_scored_results.sort(key=lambda x: x.get('risk_score', 999))
        
        # Save results
        output_file = os.path.join('data', 'direct_risk_analysis.json')
        output_data = {
            "source": "src/data/securities_data.csv",
            "risk_scored_securities": risk_scored_results
        }
        
        if save_to_json(output_data, output_file):
            logger.info(f"Saved {len(risk_scored_results)} risk-scored securities to {output_file}")
            return True
        else:
            logger.error("Failed to save results")
            return False
            
    except Exception as e:
        logger.error(f"Error in direct risk analysis: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_direct_risk_analysis() 