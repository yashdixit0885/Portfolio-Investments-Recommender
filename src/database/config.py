"""
Database configuration settings
"""

import os
from pathlib import Path

# Database paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DB_PATH = DATA_DIR / 'portfolio.db'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Database settings
DB_SETTINGS = {
    'timeout': 30,  # seconds
    'isolation_level': None,  # autocommit mode
    'detect_types': 0,
    'check_same_thread': False
}

# Table schemas
TABLE_SCHEMAS = {
    'securities': '''
        CREATE TABLE IF NOT EXISTS securities (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            industry TEXT,
            market_cap REAL,
            price REAL,
            volume REAL,
            volume_ma50 REAL,
            volume_ma200 REAL,
            ma50 REAL,
            ma200 REAL,
            beta REAL,
            rsi REAL,
            macd REAL,
            inst_own_pct REAL,
            div_yield REAL,
            last_updated TIMESTAMP
        )
    ''',
    
    'historical_data': '''
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            FOREIGN KEY (ticker) REFERENCES securities(ticker)
        )
    ''',
    
    'analysis_results': '''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            analysis_type TEXT,
            score REAL,
            metrics JSON,
            timestamp TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES securities(ticker)
        )
    '''
}

# Indexes
INDEXES = {
    'historical_data_date': '''
        CREATE INDEX IF NOT EXISTS idx_historical_data_date 
        ON historical_data(date)
    ''',
    
    'historical_data_ticker_date': '''
        CREATE INDEX IF NOT EXISTS idx_historical_data_ticker_date 
        ON historical_data(ticker, date)
    ''',
    
    'analysis_results_ticker_type': '''
        CREATE INDEX IF NOT EXISTS idx_analysis_results_ticker_type 
        ON analysis_results(ticker, analysis_type)
    '''
} 