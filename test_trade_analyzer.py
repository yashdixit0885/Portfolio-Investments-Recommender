import os
import logging
from src.trade_analyst.trade_analyzer import TradeAnalyzer
from src.investment_analyst.investment_analyzer import InvestmentAnalyzer
from src.research_analyst.research_analyzer import ResearchAnalyzer

def test_trade_analyzer():
    """Test the trade analyzer with sample data."""
    try:
        # Initialize analyzers
        investment_analyzer = InvestmentAnalyzer()
        research_analyzer = ResearchAnalyzer()
        trade_analyzer = TradeAnalyzer()
        
        # Process securities data file
        securities_file = 'data/securities_data.csv'
        if not os.path.exists(securities_file):
            print(f"Error: {securities_file} not found")
            return
            
        print("\nProcessing securities data...")
        opportunities = investment_analyzer.process_securities_data(securities_file)
        print(f"Found {len(opportunities)} initial opportunities")
        
        # Generate trade recommendations
        recommendations = investment_analyzer.generate_trade_recommendations(opportunities)
        print(f"\nGenerated {len(recommendations)} trade recommendations")
        
        # Analyze each recommendation
        for rec in recommendations[:10]:  # Show details for top 10
            ticker = rec['ticker']
            print(f"\nAnalyzing {ticker}:")
            print("--------------------------------------------------")
            
            # Analyze across different timeframes
            for timeframe in ['short_term', 'medium_term', 'long_term']:
                print(f"\nTimeframe: {timeframe}")
                print("-" * 30)
                
                analysis = trade_analyzer.analyze_swing_patterns(ticker, timeframe)
                if analysis:
                    print("Analysis Results:")
                    print(f"Current Price: ${analysis['current_price']:.2f}")
                    
                    if 'price_changes' in analysis:
                        print("\nPrice Changes:")
                        for period, change in analysis['price_changes'].items():
                            print(f"{period}: {change:+.2%}")
                    
                    if 'volume_changes' in analysis:
                        print("\nVolume Changes:")
                        for period, change in analysis['volume_changes'].items():
                            print(f"{period}: {change:+.2%}")
                    
                    if 'technical_indicators' in analysis:
                        print("\nTechnical Indicators:")
                        tech = analysis['technical_indicators']
                        
                        if 'ichimoku' in tech:
                            print("Ichimoku Cloud:")
                            for k, v in tech['ichimoku'].items():
                                print(f"  {k}: {v}")
                        
                        if 'bollinger' in tech:
                            print("\nBollinger Bands:")
                            for k, v in tech['bollinger'].items():
                                print(f"  {k}: {v}")
                        
                        if 'stochastic' in tech:
                            print("\nStochastic:")
                            for k, v in tech['stochastic'].items():
                                print(f"  {k}: {v}")
                    
                    if 'pattern_analysis' in analysis:
                        print("\nPattern Analysis:")
                        pattern = analysis['pattern_analysis']
                        print(f"Action: {pattern.get('action', 'HOLD')}")
                        print(f"Direction: {pattern['direction']}")
                        print(f"Strength: {pattern['strength']}")
                        if pattern.get('signals'):
                            print("Signals:")
                            for signal in pattern['signals']:
                                print(f"  - {signal}")
                else:
                    print("No analysis available")
        
        # Print summary of all recommendations
        print("\n" + "=" * 50)
        print("TRADE SIGNALS SUMMARY")
        print("=" * 50)
        
        for rec in recommendations[:10]:  # Show summary for top 10
            ticker = rec['ticker']
            print(f"\n{ticker}:")
            for timeframe in ['short_term', 'medium_term', 'long_term']:
                analysis = trade_analyzer.analyze_swing_patterns(ticker, timeframe)
                if analysis and 'pattern_analysis' in analysis:
                    pattern = analysis['pattern_analysis']
                    action = pattern.get('action', 'HOLD')
                    direction = pattern['direction']
                    strength = abs(pattern['strength'])
                    strength_label = 'Strong' if strength > 1 else 'Moderate' if strength == 1 else 'Neutral'
                    print(f"  {timeframe}: {action} ({strength_label}) - Current Price: ${analysis['current_price']:.2f}")
                    if pattern.get('signals'):
                        print("    Signals:")
                        for signal in pattern['signals']:
                            print(f"      - {signal}")
                else:
                    print(f"  {timeframe}: No signal available")
                    
    except Exception as e:
        print(f"Error in test_trade_analyzer: {str(e)}")

if __name__ == "__main__":
    test_trade_analyzer() 