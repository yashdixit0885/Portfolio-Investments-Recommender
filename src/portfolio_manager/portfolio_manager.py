import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from utils.common import setup_logging, get_current_time, save_to_json, load_from_json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class PortfolioManager:
    def __init__(self):
        self.logger = setup_logging('portfolio_manager')
        self.trade_analysis_file = 'data/trade_analysis.json'
        self.portfolio_file = 'data/portfolio.json'
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': os.getenv('GMAIL_USERNAME'),
            'password': os.getenv('GMAIL_PASSWORD')
        }
        
    def calculate_position_size(self, trade_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate appropriate position size based on risk parameters"""
        try:
            # Get current portfolio value
            portfolio = load_from_json(self.portfolio_file)
            total_portfolio_value = portfolio.get('total_value', 100000)  # Default to 100k if not set
            
            # Get risk metrics
            volatility = trade_analysis['risk_metrics'].get('volatility', 0.3)
            max_drawdown = trade_analysis['risk_metrics'].get('max_drawdown', -0.2)
            
            # Calculate position size based on risk
            risk_per_trade = 0.02  # 2% risk per trade
            position_size = total_portfolio_value * risk_per_trade
            
            # Adjust for volatility
            if volatility > 0.4:  # High volatility
                position_size *= 0.5
            elif volatility > 0.3:  # Medium volatility
                position_size *= 0.75
                
            # Adjust for drawdown
            if abs(max_drawdown) > 0.3:  # High drawdown
                position_size *= 0.5
            elif abs(max_drawdown) > 0.2:  # Medium drawdown
                position_size *= 0.75
                
            # Get current price and calculate shares
            current_price = trade_analysis['technical_analysis'].get('current_price', 0)
            shares = int(position_size / current_price) if current_price > 0 else 0
            
            return {
                'position_size': position_size,
                'shares': shares,
                'current_price': current_price,
                'total_value': shares * current_price
            }
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return {}

    def generate_trade_email(self, trade_analysis: Dict[str, Any], 
                           position_size: Dict[str, Any]) -> str:
        """Generate email content for trade execution"""
        ticker = trade_analysis['ticker']
        action = trade_analysis['original_recommendation']['recommendation']['action']
        shares = position_size['shares']
        price = position_size['current_price']
        
        email_content = f"""
        Trade Execution Request
        
        Ticker: {ticker}
        Action: {action.upper()}
        Number of Shares: {shares}
        Current Price: ${price:.2f}
        Total Position Value: ${position_size['total_value']:.2f}
        
        Analysis Summary:
        - Technical Score: {trade_analysis['trade_decision']['score']}
        - Confidence: {trade_analysis['trade_decision']['confidence']}
        - Volatility: {trade_analysis['risk_metrics'].get('volatility', 0):.2%}
        - Sharpe Ratio: {trade_analysis['risk_metrics'].get('sharpe_ratio', 0):.2f}
        
        Key Reasons:
        {chr(10).join('- ' + reason for reason in trade_analysis['trade_decision']['reasons'])}
        
        Please execute this trade at your earliest convenience.
        """
        
        return email_content

    def send_trade_email(self, email_content: str) -> bool:
        """Send trade execution email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['username']  # Sending to self
            msg['Subject'] = "Trade Execution Request"
            
            msg.attach(MIMEText(email_content, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], 
                                self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], 
                        self.email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info("Trade execution email sent successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending trade email: {str(e)}")
            return False

    def process_trade_analysis(self) -> None:
        """Process approved trades and generate execution instructions"""
        try:
            trade_analyses = load_from_json(self.trade_analysis_file)
            if not trade_analyses:
                self.logger.info("No new trade analyses to process")
                return
                
            for analysis in trade_analyses:
                if analysis['trade_decision']['action'] == 'execute':
                    self.logger.info(f"Processing trade for {analysis['ticker']}")
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(analysis)
                    
                    # Generate and send email
                    email_content = self.generate_trade_email(analysis, position_size)
                    if self.send_trade_email(email_content):
                        self.logger.info(f"Trade execution email sent for {analysis['ticker']}")
                    else:
                        self.logger.error(f"Failed to send trade execution email for {analysis['ticker']}")
                    
        except Exception as e:
            self.logger.error(f"Error processing trade analyses: {str(e)}")

if __name__ == "__main__":
    manager = PortfolioManager()
    manager.process_trade_analysis()