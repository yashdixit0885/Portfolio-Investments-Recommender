"""
Test suite for DatabaseManager class
"""

import unittest
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from src.database.db_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = 'data/test_portfolio.db'
        self.db = DatabaseManager(self.test_db_path)
        
        # Sample test data
        self.test_security = {
            'ticker': 'TEST',
            'name': 'Test Company',
            'industry': 'Technology',
            'market_cap': 1000000000,
            'price': 100.0,
            'volume': 1000000,
            'volume_ma50': 900000,
            'volume_ma200': 800000,
            'ma50': 95.0,
            'ma200': 90.0,
            'beta': 1.2,
            'rsi': 55.0,
            'macd': 0.5,
            'inst_own_pct': 0.75,
            'div_yield': 0.02,
            'last_updated': datetime.now().isoformat()
        }
        
        # Sample historical data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.test_historical_data = pd.DataFrame({
            'Open': [100.0] * len(dates),
            'High': [105.0] * len(dates),
            'Low': [95.0] * len(dates),
            'Close': [102.0] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)
        
        # Sample analysis result
        self.test_analysis = {
            'ticker': 'TEST',
            'analysis_type': 'technical',
            'score': 0.85,
            'metrics': {
                'rsi': 55.0,
                'macd': 0.5,
                'trend': 'bullish'
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_init_db(self):
        """Test database initialization"""
        # Verify tables exist
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall()]
            
            self.assertIn('securities', tables)
            self.assertIn('historical_data', tables)
            self.assertIn('analysis_results', tables)
    
    def test_insert_security(self):
        """Test inserting security data"""
        # Insert test security
        self.assertTrue(self.db.insert_securities([self.test_security]))
        
        # Verify data was inserted
        result = self.db.get_security('TEST')
        self.assertIsNotNone(result)
        self.assertEqual(result['ticker'], 'TEST')
        self.assertEqual(result['name'], 'Test Company')
    
    def test_insert_historical_data(self):
        """Test inserting historical data"""
        # Insert test security first
        self.db.insert_securities([self.test_security])
        
        # Insert historical data
        self.assertTrue(self.db.insert_historical_data('TEST', self.test_historical_data))
        
        # Verify data was inserted
        result = self.db.get_historical_data('TEST')
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertEqual(len(result), len(self.test_historical_data))
    
    def test_insert_analysis_result(self):
        """Test inserting analysis result"""
        # Insert test security first
        self.db.insert_securities([self.test_security])
        
        # Insert analysis result
        self.assertTrue(self.db.insert_analysis_result(
            'TEST',
            'technical',
            0.85,
            self.test_analysis['metrics']
        ))
        
        # Verify data was inserted
        result = self.db.get_latest_analysis('TEST', 'technical')
        self.assertIsNotNone(result)
        self.assertEqual(result['score'], 0.85)
        self.assertEqual(result['analysis_type'], 'technical')
    
    def test_clean_numeric(self):
        """Test numeric value cleaning"""
        test_cases = [
            ('1.5M', 1500000.0),
            ('2.3B', 2300000000.0),
            ('45%', 0.45),
            ('1,234.56', 1234.56),
            ('invalid', 0.0),
            (None, 0.0),
            (1000, 1000.0)
        ]
        
        for value, expected in test_cases:
            result = self.db._clean_numeric(value)
            self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main() 