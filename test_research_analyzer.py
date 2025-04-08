import os
import json
from src.research_analyst.research_analyzer import ResearchAnalyzer
from src.utils.common import save_to_json, load_from_json

def test_research_analyzer():
    """Test the ResearchAnalyzer class"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Initializing ResearchAnalyzer...")
    
    # Initialize analyzer
    research_analyzer = ResearchAnalyzer()
    
    # Create sample opportunities for testing
    sample_opportunities = [
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "price": 175.50,
            "market_cap": 2800000000000,
            "pe_ratio": 28.5,
            "revenue_growth_5y": 15.2,
            "profit_margin": 0.25,
            "price_change_50d": 5.2,
            "volume_change_50d": 45.3,
            "rsi": 62.4,
            "macd": 2.1
        },
        {
            "ticker": "MSFT",
            "name": "Microsoft Corporation",
            "price": 385.20,
            "market_cap": 2900000000000,
            "pe_ratio": 32.1,
            "revenue_growth_5y": 18.5,
            "profit_margin": 0.35,
            "price_change_50d": 8.7,
            "volume_change_50d": 65.2,
            "rsi": 58.9,
            "macd": 1.8
        }
    ]
    
    # Save sample opportunities to file
    opportunities_file = "data/flagged_opportunities.json"
    if save_to_json(sample_opportunities, opportunities_file):
        print(f"Saved {len(sample_opportunities)} sample opportunities to {opportunities_file}")
    else:
        print("Failed to save sample opportunities")
        return
    
    # Load opportunities from file
    loaded_opportunities = load_from_json(opportunities_file)
    if loaded_opportunities:
        print(f"Loaded {len(loaded_opportunities)} opportunities from file\n")
    else:
        print("Failed to load opportunities")
        return
    
    print("Processing opportunities...")
    research_analyzer.process_flagged_opportunities(loaded_opportunities)

if __name__ == "__main__":
    test_research_analyzer() 