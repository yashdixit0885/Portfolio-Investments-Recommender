import os
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.common import save_to_json, load_from_json, get_current_time, save_recommendations_to_csv
import numpy as np

class InvestmentAnalyzer:
    """Analyzes securities data from spreadsheets to identify key companies for research."""
    
    def __init__(self):
        """Initialize the InvestmentAnalyzer with configuration and logging."""
        self.logger = logging.getLogger('investment_analyzer')
        self.data_dir = 'data'
        self.flagged_opportunities_file = os.path.join(self.data_dir, 'flagged_opportunities.json')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load configuration
        self.load_config()
        
    def load_config(self):
        """Load configuration from environment variables or use defaults."""
        self.min_volume = int(os.getenv('MIN_VOLUME', '500000'))  # Lowered to 500K shares
        self.min_market_cap = float(os.getenv('MIN_MARKET_CAP', '50000000'))  # Lowered to $50M
        self.price_change_threshold = float(os.getenv('PRICE_CHANGE_THRESHOLD', '0.002'))  # Lowered to 0.2%
        self.volume_change_threshold = float(os.getenv('VOLUME_CHANGE_THRESHOLD', '0.05'))  # Lowered to 5%
        
    def process_securities_data(self, file_path: str) -> List[Dict]:
        """Process securities data from CSV or Excel file and identify investment opportunities."""
        try:
            # Read file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel.")

            # Log available columns for debugging
            self.logger.info(f"Available columns: {df.columns.tolist()}")

            # Function to clean and convert numeric strings
            def clean_numeric(x):
                if pd.isna(x) or x == 'N/A':
                    return 0.0
                if isinstance(x, (int, float)):
                    return float(x)
                # Remove any spaces and handle percentage signs
                x = str(x).strip().replace(' ', '')
                # Remove percentage sign and handle plus signs
                x = x.replace('%', '').replace('+', '')
                # Remove commas from numbers
                x = x.replace(',', '')
                try:
                    return float(x)
                except:
                    return 0.0

            # Convert percentage strings to floats
            percentage_columns = ['% Insider', 'Div Yield(a)', 'Sales %(q)', 'Sales %(a)', 'Profit%', '5Y Rev%', '14D Rel Str']
            for col in percentage_columns:
                if col in df.columns:
                    df[col] = df[col].apply(clean_numeric) / 100

            # Convert numeric columns
            numeric_columns = ['Last', 'Volume', 'Market Cap', 'Price Vol', '200D Avg Vol', 
                             '50D Avg Vol', '14D MACD', '200D MA', '50D MA', 'P/C OI',
                             'Mean Target', '% Institutional', 'Beta', 'Debt/Equity', 'Price/Book', 'P/E fwd']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(clean_numeric)

            opportunities = []
            for _, row in df.iterrows():
                try:
                    # Calculate volume change relative to averages (more lenient)
                    volume_change_50d = ((row['Volume'] / row['50D Avg Vol']) - 1) * 100 if pd.notnull(row['50D Avg Vol']) and row['50D Avg Vol'] > 0 else 0
                    volume_change_200d = ((row['Volume'] / row['200D Avg Vol']) - 1) * 100 if pd.notnull(row['200D Avg Vol']) and row['200D Avg Vol'] > 0 else 0

                    # Calculate price momentum (more lenient)
                    price_momentum_50d = ((row['Last'] / row['50D MA']) - 1) * 100 if pd.notnull(row['50D MA']) and row['50D MA'] > 0 else 0
                    price_momentum_200d = ((row['Last'] / row['200D MA']) - 1) * 100 if pd.notnull(row['200D MA']) and row['200D MA'] > 0 else 0

                    # Enhanced opportunity criteria (more lenient)
                    if (row['Volume'] >= self.min_volume * 0.8 or  # 20% more lenient on volume
                        row['Market Cap'] >= self.min_market_cap * 0.8 or  # 20% more lenient on market cap
                        abs(price_momentum_50d) >= self.price_change_threshold * 0.8 or  # 20% more lenient on price change
                        volume_change_50d >= self.volume_change_threshold * 0.8 or  # 20% more lenient on volume change
                        volume_change_200d >= self.volume_change_threshold * 0.8):  # Consider 200-day volume change
                        
                        # Calculate opportunity score with new metrics
                        opportunity = {
                            'ticker': row['Symbol'],
                            'name': row['Name'],
                            'price': row['Last'],
                            'volume': row['Volume'],
                            'market_cap': row['Market Cap'],
                            'price_change_50d': price_momentum_50d,
                            'price_change_200d': price_momentum_200d,
                            'volume_change_50d': volume_change_50d,
                            'volume_change_200d': volume_change_200d,
                            'industry': row['Industry'],
                            'sector': row['Sector'],
                            'beta': row['Beta'],
                            'pe_ratio': row['P/E fwd'],
                            'price_to_book': row['Price/Book'],
                            'debt_to_equity': row['Debt/Equity'],
                            'profit_margin': row['Profit%'],
                            'revenue_growth_5y': row['5Y Rev%'],
                            'sales_growth_q': row['Sales %(q)'],
                            'sales_growth_a': row['Sales %(a)'],
                            'insider_ownership': row['% Insider'],
                            'institutional_ownership': row['% Institutional'],
                            'dividend_yield': row['Div Yield(a)'],
                            'analyst_rating': row['Analyst Rating'],
                            'mean_target': row['Mean Target'],
                            'macd': row['14D MACD'],
                            'rsi': row['14D Rel Str'],
                            'timestamp': get_current_time().isoformat(),
                            'score': self._calculate_opportunity_score({
                                'price_momentum_50d': price_momentum_50d,
                                'price_momentum_200d': price_momentum_200d,
                                'volume_change_50d': volume_change_50d,
                                'volume_change_200d': volume_change_200d,
                                'market_cap': row['Market Cap'],
                                'beta': row['Beta'],
                                'pe_ratio': row['P/E fwd'],
                                'profit_margin': row['Profit%'],
                                'revenue_growth': row['5Y Rev%'],
                                'analyst_rating': row['Analyst Rating'],
                                'rsi': row['14D Rel Str']
                            })
                        }
                        opportunities.append(opportunity)

                except Exception as e:
                    self.logger.warning(f"Error processing row for {row.get('Symbol', 'Unknown')}: {str(e)}")
                    continue

            # Sort opportunities by score
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            
            self.logger.info(f"Found {len(opportunities)} investment opportunities")
            return opportunities

        except Exception as e:
            self.logger.error(f"Error processing securities data: {str(e)}")
            return []

    def _calculate_opportunity_score(self, metrics: Dict) -> float:
        """Calculate a composite score for an investment opportunity."""
        try:
            # Price momentum component (30%)
            momentum_score = (
                (max(0, metrics['price_momentum_50d']) * 0.6) +  # Short-term momentum
                (max(0, metrics['price_momentum_200d']) * 0.4)   # Long-term momentum
            ) / 100  # Normalize to 0-1 range
            
            # Volume surge component (20%)
            volume_score = (
                (max(0, metrics['volume_change_50d']) * 0.6) +   # Short-term volume
                (max(0, metrics['volume_change_200d']) * 0.4)     # Long-term volume
            ) / 200  # Normalize to 0-1 range
            
            # Market cap tier component (10%)
            market_cap = metrics['market_cap']
            if market_cap >= 200e9:  # Mega cap
                cap_score = 1.0
            elif market_cap >= 10e9:  # Large cap
                cap_score = 0.8
            elif market_cap >= 2e9:   # Mid cap
                cap_score = 0.6
            elif market_cap >= 1e9:   # Small cap
                cap_score = 0.4
            else:
                cap_score = 0.2
            
            # Fundamental component (20%)
            fundamental_score = 0.0
            if metrics['pe_ratio'] and metrics['pe_ratio'] > 0:
                pe_score = min(1.0, 30 / metrics['pe_ratio'])  # Lower P/E is better
                fundamental_score += pe_score * 0.3
            
            if metrics['profit_margin']:
                margin_score = min(1.0, metrics['profit_margin'] / 20)  # 20% margin is considered excellent
                fundamental_score += margin_score * 0.3
            
            if metrics['revenue_growth']:
                growth_score = min(1.0, metrics['revenue_growth'] / 30)  # 30% growth is considered excellent
                fundamental_score += growth_score * 0.4
            
            # Technical component (10%)
            technical_score = 0.0
            if metrics['rsi']:
                rsi = metrics['rsi']
                if 30 <= rsi <= 70:  # Healthy range
                    technical_score += 0.5
                elif rsi < 30:  # Oversold
                    technical_score += 0.8
                elif rsi > 70:  # Overbought
                    technical_score += 0.2
            
            # Analyst sentiment component (10%)
            sentiment_score = 0.0
            if metrics['analyst_rating']:
                rating = str(metrics['analyst_rating']).upper()
                if 'STRONG BUY' in rating:
                    sentiment_score = 1.0
                elif 'BUY' in rating:
                    sentiment_score = 0.8
                elif 'HOLD' in rating:
                    sentiment_score = 0.5
                elif 'SELL' in rating:
                    sentiment_score = 0.2
                elif 'STRONG SELL' in rating:
                    sentiment_score = 0.0
            
            # Calculate final score with weights
            final_score = (
                momentum_score * 0.30 +      # Price momentum
                volume_score * 0.20 +        # Volume surge
                cap_score * 0.10 +           # Market cap tier
                fundamental_score * 0.20 +   # Fundamentals
                technical_score * 0.10 +     # Technical indicators
                sentiment_score * 0.10       # Analyst sentiment
            )
            
            return round(final_score, 4)
            
        except Exception as e:
            self.logger.warning(f"Error calculating opportunity score: {str(e)}")
            return 0.0
    
    def analyze_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze opportunities and rank them based on various metrics.
        
        Args:
            opportunities: List of opportunity dictionaries
            
        Returns:
            List of analyzed and ranked opportunities
        """
        try:
            analyzed_opportunities = []
            
            for opp in opportunities:
                # Calculate opportunity score
                score = self._calculate_opportunity_score(opp)
                
                # Add analysis results
                analyzed_opp = opp.copy()
                analyzed_opp['analysis'] = {
                    'score': score,
                    'metrics': {
                        'volume_significance': self._calculate_volume_significance(opp),
                        'price_momentum': self._calculate_price_momentum(opp),
                        'market_cap_significance': self._calculate_market_cap_significance(opp)
                    }
                }
                
                analyzed_opportunities.append(analyzed_opp)
            
            # Sort opportunities by score
            analyzed_opportunities.sort(key=lambda x: x['analysis']['score'], reverse=True)
            
            return analyzed_opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing opportunities: {str(e)}")
            return opportunities
    
    def _calculate_volume_significance(self, opportunity: Dict[str, Any]) -> float:
        """Calculate volume significance score."""
        try:
            volume = opportunity.get('volume', 0)
            volume_change = opportunity.get('volume_change', 0)
            
            # Normalize volume
            volume_score = min(volume / self.min_volume, 1.0)
            
            # Normalize volume change
            change_score = min(volume_change / self.volume_change_threshold, 1.0)
            
            # Combine scores
            return (volume_score * 0.6 + change_score * 0.4)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume significance: {str(e)}")
            return 0.0
    
    def _calculate_price_momentum(self, opportunity: Dict[str, Any]) -> float:
        """Calculate price momentum score."""
        try:
            price_change = abs(opportunity.get('price_change', 0))
            
            # Normalize price change
            return min(price_change / self.price_change_threshold, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating price momentum: {str(e)}")
            return 0.0
    
    def _calculate_market_cap_significance(self, opportunity: Dict[str, Any]) -> float:
        """Calculate market cap significance score."""
        try:
            market_cap = opportunity.get('market_cap', 0)
            
            # Normalize market cap
            return min(market_cap / self.min_market_cap, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating market cap significance: {str(e)}")
            return 0.0
    
    def save_opportunities(self, opportunities: List[Dict[str, Any]]) -> bool:
        """
        Save opportunities to a JSON file.
        
        Args:
            opportunities: List of opportunity dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing opportunities
            existing_opportunities = []
            if os.path.exists(self.flagged_opportunities_file):
                existing_opportunities = load_from_json(self.flagged_opportunities_file) or []
            
            # Combine with new opportunities
            all_opportunities = existing_opportunities + opportunities
            
            # Remove duplicates based on ticker and timestamp
            seen = set()
            unique_opportunities = []
            for opp in all_opportunities:
                # Skip opportunities without required fields
                if not all(key in opp for key in ['ticker', 'timestamp']):
                    self.logger.warning(f"Skipping opportunity missing required fields: {opp}")
                    continue
                    
                key = (opp['ticker'], opp['timestamp'])
                if key not in seen:
                    seen.add(key)
                    unique_opportunities.append(opp)
            
            if not unique_opportunities:
                self.logger.warning("No valid opportunities to save")
                return False
            
            # Save to file
            if save_to_json(unique_opportunities, self.flagged_opportunities_file):
                self.logger.info(f"Saved {len(unique_opportunities)} opportunities to {self.flagged_opportunities_file}")
                return True
            else:
                self.logger.error("Failed to save opportunities")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving opportunities: {str(e)}")
            return False
    
    def process_securities_file(self, file_path: str) -> bool:
        """
        Process a securities data file and save opportunities.
        
        Args:
            file_path: Path to the securities data file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Process securities data
            opportunities = self.process_securities_data(file_path)
            
            if not opportunities:
                self.logger.warning("No opportunities found in securities data")
                return False
            
            # Analyze opportunities
            analyzed_opportunities = self.analyze_opportunities(opportunities)
            
            if not analyzed_opportunities:
                self.logger.warning("No analyzed opportunities found")
                return False
            
            # Save analyzed opportunities
            return self.save_opportunities(analyzed_opportunities)
            
        except Exception as e:
            self.logger.error(f"Error processing securities file: {str(e)}")
            return False
    
    def generate_trade_recommendations(self, opportunities, max_recommendations=None):
        """Generate trade recommendations from investment opportunities."""
        try:
            if not opportunities:
                self.logger.warning("No opportunities to generate recommendations from")
                return []
                
            # Calculate minimum recommendations (1% of total opportunities)
            min_recommendations = max(int(len(opportunities) * 0.01), 38)  # At least 38 recommendations (1% of 3800+ stocks)
            if max_recommendations is None:
                max_recommendations = max(min_recommendations, 100)  # At least min_recommendations or 100
                
            # Sort opportunities by score
            sorted_opportunities = sorted(opportunities, key=lambda x: x['score'], reverse=True)
            
            # Take top opportunities up to max_recommendations
            recommendations = []
            for opportunity in sorted_opportunities[:max_recommendations]:
                # More lenient criteria for recommendations
                price_momentum = opportunity.get('price_change_50d', 0)
                volume_change = opportunity.get('volume_change_50d', 0)
                rsi = opportunity.get('rsi', 50)
                macd = opportunity.get('macd', 0)
                score = opportunity.get('score', 0)
                
                # Lowered thresholds for action determination
                if (price_momentum > -2 or  # Allow slightly negative momentum
                    (rsi < 45 and volume_change > -5) or  # More lenient RSI and volume
                    macd > -2 or  # Allow slightly negative MACD
                    score > 0.3):  # Lower score threshold
                    action = 'BUY'
                else:
                    action = 'SELL'
                
                # Calculate position size based on market cap and score
                position_size = self._calculate_position_size(opportunity['market_cap'])
                position_size = min(position_size * (1 + opportunity['score']), 0.15)
                
                recommendation = {
                    'ticker': opportunity['ticker'],
                    'name': opportunity['name'],
                    'action': action,
                    'score': opportunity['score'],
                    'position_size': round(position_size, 3),
                    'current_price': opportunity['price'],
                    'market_cap': opportunity['market_cap'],
                    'price_change': price_momentum,
                    'volume_change': volume_change,
                    'rationale': self._generate_rationale(opportunity)
                }
                recommendations.append(recommendation)
            
            # Ensure minimum number of recommendations
            if len(recommendations) < min_recommendations:
                self.logger.warning(f"Generated only {len(recommendations)} recommendations, below minimum of {min_recommendations}")
                # Add more recommendations from remaining opportunities if needed
                remaining_opportunities = sorted_opportunities[len(recommendations):2*max_recommendations]
                for opportunity in remaining_opportunities:
                    if len(recommendations) >= min_recommendations:
                        break
                    recommendation = {
                        'ticker': opportunity['ticker'],
                        'name': opportunity['name'],
                        'action': 'BUY' if opportunity.get('score', 0) > 0.2 else 'SELL',
                        'score': opportunity['score'],
                        'position_size': round(self._calculate_position_size(opportunity['market_cap']), 3),
                        'current_price': opportunity['price'],
                        'market_cap': opportunity['market_cap'],
                        'price_change': opportunity.get('price_change_50d', 0),
                        'volume_change': opportunity.get('volume_change_50d', 0),
                        'rationale': self._generate_rationale(opportunity)
                    }
                    recommendations.append(recommendation)
            
            # Save recommendations to CSV file
            csv_filepath = save_recommendations_to_csv(recommendations)
            self.logger.info(f"Saved recommendations to CSV file: {csv_filepath}")
            
            # Also save to JSON for backward compatibility
            save_to_json(recommendations, 'data/trade_recommendations.json')
            
            self.logger.info(f"Generated {len(recommendations)} trade recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating trade recommendations: {str(e)}")
            return []
        
    def _generate_rationale(self, opportunity: Dict) -> str:
        """Generate a rationale string for the trade recommendation."""
        try:
            rationale_parts = []
            
            # Price momentum
            if opportunity['price_change_50d'] > 0:
                rationale_parts.append(f"Strong upward price momentum ({opportunity['price_change_50d']:.1f}% over 50 days)")
            else:
                rationale_parts.append(f"Downward price trend ({opportunity['price_change_50d']:.1f}% over 50 days)")
                
            # Volume analysis
            if opportunity['volume_change_50d'] > 100:
                rationale_parts.append(f"Exceptional volume surge ({opportunity['volume_change_50d']:.1f}% above 50-day average)")
            elif opportunity['volume_change_50d'] > 50:
                rationale_parts.append(f"Strong volume increase ({opportunity['volume_change_50d']:.1f}% above 50-day average)")
            
            # Technical indicators
            if opportunity.get('rsi') > 70:
                rationale_parts.append("Overbought conditions (RSI > 70)")
            elif opportunity.get('rsi') < 30:
                rationale_parts.append("Oversold conditions (RSI < 30)")
            
            if opportunity.get('macd') and opportunity.get('macd') > 0:
                rationale_parts.append("Positive MACD signal")
            
            # Fundamental factors
            if opportunity.get('pe_ratio') and opportunity['pe_ratio'] > 0:
                if opportunity['pe_ratio'] < 15:
                    rationale_parts.append("Attractive valuation (low P/E ratio)")
                elif opportunity['pe_ratio'] > 50:
                    rationale_parts.append("High valuation (elevated P/E ratio)")
            
            if opportunity.get('revenue_growth_5y') and opportunity['revenue_growth_5y'] > 20:
                rationale_parts.append(f"Strong revenue growth ({opportunity['revenue_growth_5y']:.1f}% over 5 years)")
            
            if opportunity.get('profit_margin') and opportunity['profit_margin'] > 0.15:
                rationale_parts.append(f"Healthy profit margins ({opportunity['profit_margin']*100:.1f}%)")
            
            # Market cap consideration
            market_cap_b = opportunity['market_cap'] / 1_000_000_000
            if market_cap_b >= 200:
                rationale_parts.append(f"Mega-cap stock (${market_cap_b:.1f}B market cap)")
            elif market_cap_b >= 10:
                rationale_parts.append(f"Large-cap stock (${market_cap_b:.1f}B market cap)")
            elif market_cap_b >= 2:
                rationale_parts.append(f"Mid-cap stock (${market_cap_b:.1f}B market cap)")
            else:
                rationale_parts.append(f"Small-cap stock (${market_cap_b:.1f}B market cap)")
            
            # Combine all rationale parts
            rationale = " | ".join(rationale_parts)
            
            return rationale
            
        except Exception as e:
            self.logger.warning(f"Error generating rationale for {opportunity.get('ticker', 'unknown')}: {str(e)}")
            return "Technical and fundamental factors indicate a trading opportunity."

    def _calculate_position_size(self, market_cap: float) -> float:
        """Calculate position size based on market cap."""
        try:
            if market_cap >= 10_000_000_000:  # >$10B
                base_size = 0.10  # Up to 10% of portfolio
            elif market_cap >= 1_000_000_000:  # >$1B
                base_size = 0.05  # Up to 5% of portfolio
            elif market_cap >= 500_000_000:  # >$500M
                base_size = 0.03  # Up to 3% of portfolio
            else:
                base_size = 0.02  # Up to 2% of portfolio
                
            return round(base_size, 4)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.02  # Default to 2% if there's an error 