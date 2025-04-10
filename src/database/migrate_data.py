"""
Data Migration Script

This script migrates existing CSV data to the SQLite database.
"""

import os
import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import yfinance as yf
from src.database.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_migration')

def load_securities_data(csv_path: str) -> pd.DataFrame:
    """Load securities data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} securities from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading securities data: {str(e)}")
        raise

def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        logger.info(f"Successfully fetched historical data for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return pd.DataFrame()

def insert_historical_data_to_db(db_manager, ticker, data):
    """Insert historical data into the database."""
    if data is not None and not data.empty:
        # Ensure columns are lowercase before inserting
        data.columns = [col.lower() for col in data.columns]
        if 'open' in data.columns: # Check if necessary columns exist after lowercasing
            return db_manager.insert_historical_data(ticker, data)
        else:
            logger.error(f"Missing required columns (e.g., 'open') after lowercasing for {ticker}")
            return False
    return False

def migrate_data():
    """Migrate data from CSV to database."""
    try:
        # Initialize database manager
        db = DatabaseManager()
        
        # Load securities data
        csv_path = 'data/securities_data.csv'
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return
            
        df = load_securities_data(csv_path)
        
        # Convert DataFrame to list of dictionaries
        securities = df.to_dict('records')
        
        # Insert securities data
        if db.insert_securities(securities):
            logger.info("Successfully migrated securities data")
            
            # Fetch and insert historical data for each security
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            for security in securities:
                ticker = security.get('Symbol', '').upper()
                if not ticker:
                    continue
                    
                historical_data = fetch_historical_data(ticker, start_date, end_date)
                if not historical_data.empty:
                    insert_historical_data_to_db(db, ticker, historical_data)
                    
        else:
            logger.error("Failed to migrate securities data")
            
    except Exception as e:
        logger.error(f"Error during data migration: {str(e)}")
        raise

if __name__ == '__main__':
    migrate_data() 