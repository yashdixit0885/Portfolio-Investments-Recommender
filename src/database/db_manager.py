"""
Database Manager for Portfolio Investments Recommender

This module handles all database operations for the application, including:
- Creating and maintaining the database schema
- Inserting and updating securities data
- Querying securities and analysis results
- Managing historical data
"""

import os
import sqlite3
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from .config import DB_PATH, DB_SETTINGS, TABLE_SCHEMAS, INDEXES

class DatabaseManager:
    """Manages all database operations for the application."""
    
    def __init__(self, db_path: str = None):
        """Initialize the database manager."""
        self.logger = logging.getLogger('database_manager')
        self.db_path = db_path or str(DB_PATH)
        
        # Initialize database
        self._init_db()
        
    def _init_db(self):
        """Initialize the database with required tables and indexes."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                cursor = conn.cursor()
                
                # Create tables
                for table_name, schema in TABLE_SCHEMAS.items():
                    cursor.execute(schema)
                    self.logger.info(f"Created table: {table_name}")
                
                # Create indexes
                for index_name, index_sql in INDEXES.items():
                    cursor.execute(index_sql)
                    self.logger.info(f"Created index: {index_name}")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
            
    def insert_securities(self, securities: List[Dict[str, Any]]) -> bool:
        """Insert or update securities data in the database."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                cursor = conn.cursor()
                
                for security in securities:
                    # Convert ticker to uppercase for consistency
                    ticker = security.get('ticker', security.get('Ticker', security.get('Symbol', ''))).upper()
                    if not ticker:
                        continue
                        
                    # Prepare data for insertion
                    data = {
                        'ticker': ticker,
                        'name': security.get('name', security.get('Name', '')),
                        'industry': security.get('industry', security.get('Industry', '')),
                        'market_cap': self._clean_numeric(security.get('market_cap', security.get('Market Cap', 0))),
                        'price': self._clean_numeric(security.get('price', security.get('Last', 0))),
                        'volume': self._clean_numeric(security.get('volume', security.get('Volume', 0))),
                        'volume_ma50': self._clean_numeric(security.get('volume_ma50', security.get('50D Avg Vol', 0))),
                        'volume_ma200': self._clean_numeric(security.get('volume_ma200', security.get('200D Avg Vol', 0))),
                        'ma50': self._clean_numeric(security.get('ma50', security.get('50D MA', 0))),
                        'ma200': self._clean_numeric(security.get('ma200', security.get('200D MA', 0))),
                        'beta': self._clean_numeric(security.get('beta', security.get('Beta', 0))),
                        'rsi': self._clean_numeric(security.get('rsi', security.get('RSI', 0))),
                        'macd': self._clean_numeric(security.get('macd', security.get('MACD', 0))),
                        'inst_own_pct': self._clean_numeric(security.get('inst_own_pct', security.get('Inst Own %', 0))),
                        'div_yield': self._clean_numeric(security.get('div_yield', security.get('Div Yield', 0))),
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    # Insert or update
                    cursor.execute('''
                        INSERT OR REPLACE INTO securities 
                        (ticker, name, industry, market_cap, price, volume, volume_ma50, 
                         volume_ma200, ma50, ma200, beta, rsi, macd, inst_own_pct, 
                         div_yield, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', tuple(data.values()))
                    
                conn.commit()
                self.logger.info(f"Successfully inserted/updated {len(securities)} securities")
                return True
                
        except Exception as e:
            self.logger.error(f"Error inserting securities: {str(e)}")
            return False
            
    def insert_historical_data(self, ticker: str, data: pd.DataFrame) -> bool:
        """Insert historical price data for a security."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                cursor = conn.cursor()
                
                # Convert column names to lowercase to handle different case formats
                data_processed = data.copy()
                data_processed.columns = data_processed.columns.str.lower()
                
                # Convert DataFrame to list of tuples
                records = []
                for index, row in data_processed.iterrows():
                    records.append((
                        ticker.upper(),
                        index.date().isoformat(), 
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row['volume']
                    ))
                
                # Insert in batches
                cursor.executemany('''
                    INSERT INTO historical_data 
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', records)
                
                conn.commit()
                self.logger.info(f"Successfully inserted {len(records)} historical records for {ticker}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error inserting historical data for {ticker}: {str(e)}")
            return False
            
    def insert_analysis_result(self, ticker: str, analysis_type: str, 
                             score: float, metrics: Dict[str, Any]) -> bool:
        """Insert analysis results for a security."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO analysis_results 
                    (ticker, analysis_type, score, metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    ticker.upper(),
                    analysis_type,
                    score,
                    json.dumps(metrics),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                self.logger.info(f"Successfully inserted analysis result for {ticker}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error inserting analysis result for {ticker}: {str(e)}")
            return False
            
    def get_security(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get security data by ticker."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM securities WHERE ticker = ?', (ticker.upper(),))
                row = cursor.fetchone()
                
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting security {ticker}: {str(e)}")
            return None
            
    def get_historical_data(self, ticker: str, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get historical data for a security."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                query = 'SELECT * FROM historical_data WHERE ticker = ?'
                params = [ticker.upper()]
                
                if start_date:
                    query += ' AND date >= ?'
                    params.append(start_date)
                if end_date:
                    query += ' AND date <= ?'
                    params.append(end_date)
                    
                query += ' ORDER BY date'
                
                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            return None
            
    def get_latest_analysis(self, ticker: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest analysis result for a security."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    WHERE ticker = ? AND analysis_type = ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (ticker.upper(), analysis_type))
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    result = dict(zip(columns, row))
                    result['metrics'] = json.loads(result['metrics'])
                    return result
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting analysis for {ticker}: {str(e)}")
            return None
            
    def get_all_securities(self) -> List[Dict[str, Any]]:
        """Get all securities from the database."""
        try:
            with sqlite3.connect(self.db_path, **DB_SETTINGS) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM securities')
                rows = cursor.fetchall()
                
                if rows:
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting all securities: {str(e)}")
            return []
            
    def _clean_numeric(self, value: Any) -> float:
        """Clean and convert numeric values from various formats."""
        if pd.isna(value) or value is None:
            return 0.0
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Remove commas and convert to lowercase
            value = value.replace(',', '').lower()
            
            # Handle percentage values
            if '%' in value:
                return float(value.replace('%', '')) / 100
                
            # Handle suffixes (K, M, B, T)
            multipliers = {
                'k': 1e3,
                'm': 1e6,
                'b': 1e9,
                't': 1e12
            }
            
            for suffix, multiplier in multipliers.items():
                if value.endswith(suffix):
                    return float(value[:-1]) * multiplier
                    
            # Handle regular numbers
            try:
                return float(value)
            except ValueError:
                return 0.0
                
        return 0.0 