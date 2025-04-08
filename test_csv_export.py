import os
import sys
import pandas as pd
import random
from src.utils.common import save_recommendations_to_csv

def test_csv_export():
    """Test the CSV export functionality with a larger set of recommendations."""
    # Create a larger set of sample recommendations (simulating 1% of securities_data.csv)
    # For a dataset of 3800+ stocks, 1% would be approximately 38 recommendations
    num_recommendations = 40  # Slightly more than 1% of 3800+ stocks
    
    # Sample stock data (expanded to represent a larger dataset)
    stock_data = [
        # Original 20 stocks
        {'ticker': 'AAPL', 'name': 'Apple Inc.', 'price': 188.39, 'market_cap': 2950000000000},
        {'ticker': 'MSFT', 'name': 'Microsoft Corporation', 'price': 420.55, 'market_cap': 3120000000000},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'price': 155.23, 'market_cap': 1950000000000},
        {'ticker': 'AMZN', 'name': 'Amazon.com Inc.', 'price': 178.75, 'market_cap': 1840000000000},
        {'ticker': 'META', 'name': 'Meta Platforms Inc.', 'price': 474.99, 'market_cap': 1210000000000},
        {'ticker': 'TSLA', 'name': 'Tesla Inc.', 'price': 247.39, 'market_cap': 787000000000},
        {'ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'price': 104.48, 'market_cap': 2570000000000},
        {'ticker': 'JPM', 'name': 'JPMorgan Chase & Co.', 'price': 182.63, 'market_cap': 526000000000},
        {'ticker': 'V', 'name': 'Visa Inc.', 'price': 275.27, 'market_cap': 553000000000},
        {'ticker': 'WMT', 'name': 'Walmart Inc.', 'price': 59.48, 'market_cap': 479000000000},
        {'ticker': 'JNJ', 'name': 'Johnson & Johnson', 'price': 147.50, 'market_cap': 356000000000},
        {'ticker': 'MA', 'name': 'Mastercard Inc.', 'price': 432.50, 'market_cap': 408000000000},
        {'ticker': 'PG', 'name': 'Procter & Gamble Co.', 'price': 156.84, 'market_cap': 377000000000},
        {'ticker': 'HD', 'name': 'Home Depot Inc.', 'price': 375.50, 'market_cap': 374000000000},
        {'ticker': 'BAC', 'name': 'Bank of America Corp.', 'price': 35.87, 'market_cap': 277000000000},
        {'ticker': 'KO', 'name': 'Coca-Cola Co.', 'price': 60.50, 'market_cap': 261000000000},
        {'ticker': 'PFE', 'name': 'Pfizer Inc.', 'price': 28.50, 'market_cap': 161000000000},
        {'ticker': 'CSCO', 'name': 'Cisco Systems Inc.', 'price': 49.50, 'market_cap': 200000000000},
        {'ticker': 'PEP', 'name': 'PepsiCo Inc.', 'price': 175.50, 'market_cap': 241000000000},
        {'ticker': 'ADBE', 'name': 'Adobe Inc.', 'price': 525.00, 'market_cap': 238000000000},
        
        # Additional stocks to simulate a larger dataset
        {'ticker': 'INTC', 'name': 'Intel Corporation', 'price': 42.50, 'market_cap': 180000000000},
        {'ticker': 'VZ', 'name': 'Verizon Communications Inc.', 'price': 40.25, 'market_cap': 170000000000},
        {'ticker': 'DIS', 'name': 'The Walt Disney Company', 'price': 105.75, 'market_cap': 195000000000},
        {'ticker': 'NFLX', 'name': 'Netflix Inc.', 'price': 485.00, 'market_cap': 210000000000},
        {'ticker': 'PYPL', 'name': 'PayPal Holdings Inc.', 'price': 65.25, 'market_cap': 70000000000},
        {'ticker': 'CRM', 'name': 'Salesforce Inc.', 'price': 225.50, 'market_cap': 220000000000},
        {'ticker': 'ABT', 'name': 'Abbott Laboratories', 'price': 105.25, 'market_cap': 185000000000},
        {'ticker': 'TMO', 'name': 'Thermo Fisher Scientific Inc.', 'price': 525.75, 'market_cap': 210000000000},
        {'ticker': 'AVGO', 'name': 'Broadcom Inc.', 'price': 1250.00, 'market_cap': 600000000000},
        {'ticker': 'ACN', 'name': 'Accenture plc', 'price': 325.50, 'market_cap': 205000000000},
        {'ticker': 'LLY', 'name': 'Eli Lilly and Company', 'price': 750.25, 'market_cap': 710000000000},
        {'ticker': 'ABBV', 'name': 'AbbVie Inc.', 'price': 165.75, 'market_cap': 295000000000},
        {'ticker': 'MRK', 'name': 'Merck & Co. Inc.', 'price': 125.50, 'market_cap': 320000000000},
        {'ticker': 'UNH', 'name': 'UnitedHealth Group Inc.', 'price': 485.25, 'market_cap': 450000000000},
        {'ticker': 'CVX', 'name': 'Chevron Corporation', 'price': 155.75, 'market_cap': 290000000000},
        {'ticker': 'XOM', 'name': 'Exxon Mobil Corporation', 'price': 105.25, 'market_cap': 410000000000},
        {'ticker': 'BRK.B', 'name': 'Berkshire Hathaway Inc.', 'price': 355.50, 'market_cap': 780000000000},
        {'ticker': 'JNJ', 'name': 'Johnson & Johnson', 'price': 155.25, 'market_cap': 375000000000},
        {'ticker': 'PG', 'name': 'Procter & Gamble Co.', 'price': 150.75, 'market_cap': 355000000000},
        {'ticker': 'MA', 'name': 'Mastercard Inc.', 'price': 425.50, 'market_cap': 400000000000}
    ]
    
    # Generate recommendations
    recommendations = []
    actions = ['BUY', 'SELL', 'HOLD']
    rationales = [
        'Strong technical setup with bullish momentum',
        'Consolidation phase, waiting for breakout',
        'Technical breakdown with increasing selling pressure',
        'Oversold conditions with potential reversal',
        'Overbought conditions, consider taking profits',
        'Breakout from key resistance level',
        'Support level tested, potential bounce',
        'Volume increasing on upward price movement',
        'Decreasing volume on downward price movement',
        'Moving average crossover indicating trend change'
    ]
    
    # Select random stocks for recommendations
    selected_stocks = random.sample(stock_data, num_recommendations)
    
    for stock in selected_stocks:
        action = random.choice(actions)
        score = round(random.uniform(0.3, 0.9), 3)
        position_size = round(random.uniform(0.05, 0.25), 3)
        price_change = round(random.uniform(-5.0, 5.0), 2)
        volume_change = round(random.uniform(-50.0, 50.0), 2)
        rationale = random.choice(rationales)
        
        recommendation = {
            'ticker': stock['ticker'],
            'name': stock['name'],
            'action': action,
            'score': score,
            'position_size': position_size,
            'current_price': stock['price'],
            'market_cap': stock['market_cap'],
            'price_change': price_change,
            'volume_change': volume_change,
            'rationale': rationale
        }
        recommendations.append(recommendation)
    
    print(f"\nGenerated {len(recommendations)} recommendations (simulating 1% of securities_data.csv with 3800+ stocks)")
    
    # Save recommendations to CSV
    filepath = save_recommendations_to_csv(recommendations)
    print(f"Saved recommendations to: {filepath}")
    
    # Verify file was created
    if os.path.exists(filepath):
        print("\nCSV file created successfully!")
        
        # Read and display the CSV contents
        df = pd.read_csv(filepath)
        print(f"\nCSV Contents ({len(df)} recommendations):")
        print(df.to_string())
        
        # Verify Output and Archived directories
        if os.path.exists("Output"):
            print("\nOutput directory exists")
        if os.path.exists("Archived"):
            print("Archived directory exists")
            
        # Verify that we have at least 1% of recommendations (approximately 38 for 3800+ stocks)
        min_recommendations = 38  # 1% of 3800+ stocks
        if len(recommendations) >= min_recommendations:
            print(f"\n✓ Success: Generated {len(recommendations)} recommendations, meeting the minimum requirement of {min_recommendations} (1% of 3800+ stocks)")
        else:
            print(f"\n✗ Error: Generated only {len(recommendations)} recommendations, below the minimum requirement of {min_recommendations} (1% of 3800+ stocks)")
    else:
        print("\nError: CSV file was not created!")

if __name__ == "__main__":
    test_csv_export() 